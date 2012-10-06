//
// ribes-c
//  Copyright (C) 2012  Graham Neubig
// (original Python code is)
//  Copyright (C) 2011  Nippon Telegraph and Telephone Corporation
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
//
// This program calculates the "RIBES" machine translation evaluation measure,
// which is particularly useful for translation between language pairs with
// large amounts of reordering.
// 
// Reference:
//  Hideki Isozaki, Tsutomu Hirao, Kevin Duh, Katsuhito Sudoh, and Hajime Tsukada,
//  "Automatic Evaluation of Translation Quality for Distant Language Pairs",
//  Proceedings of the 2010 Conference on Empirical Methods in Natural Language Processing (EMNLP),
//  pp. 944--952 Cambridge MA, October, 2010
//  -- http://aclweb.org/anthology-new/D/D10/D10-1092.pdf

#include <boost/unordered_map.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <fstream>

using namespace std;
using namespace boost;
namespace po = boost::program_options;

class TauScore {
public:
    TauScore(double n, double p, double b) :
        nkt(n), precision(p), bp(b) { }
    double nkt;
    double precision;
    double bp;
};


// """Corpus class.
// 
// Stores sentences and is used for evaluation.
// 
// Attributes (private):
//     sentence_ : list of sentences (word lists)
//     numwords_ : #words in the corpus (currently not used but can be used for corpus statistics.)
// 
// Attributes (public):
//     filename   : corpus file name (set as public for error messages about the corpus)
// """
class Corpus {
public:

    // """Constructor.
    // Initialize a Corpus instance by a corpus file with a certain encoding (default utf-8).
    // Argument:
    //     _file : corpus file of "sentence-per-line" format
    // Keyword:
    //     case     : preserve uppercase letters or not, default: False
    // """
    Corpus(const string & file, bool preserve_case=false) :
            numwords_(0) {

        filename_ = file;
        ifstream fp(file.c_str());
        string line;
        while(getline(fp, line)) {
            // Remove leading/trailing whitespace
            boost::trim_if(line, boost::is_any_of(" \t"));
            // lowercasing if case is False
            if(!preserve_case)
                boost::algorithm::to_lower(line);
            // split the sentence to a word list
            vector<string> strs;
            split(strs, line, is_any_of("\t "));
            // push_back it to the corpus sentence list
            sentence_.push_back( strs );
            // count the words
            numwords_ += strs.size();
        }
    }

    int size() const { return sentence_.size(); }
    const vector<string> & operator[](size_t i) const { return sentence_[i]; }
    vector<string> & operator[](size_t i) { return sentence_[i]; }
    const string & filename() const { return filename_; }

protected:
    string filename_;
    vector< vector<string> > sentence_;
    int numwords_;

};

// """RIBES evaluator class.
// 
// Receives "Corpus" instances and score them with hyperparameters alpha and beta.
// 
// Attributes (private):
//     sent_   : show sentence-level scores or not
//     alpha_  : hyperparameter alpha, for (unigram_precision)**alpha
//     beta_   : hyperparameter beta,  for (brevity_penalty)**beta
//     output_ : output file name
// """
class RibesEvaluator {
public:

    // """Constructor.
    // 
    // Initialize a RIBESevaluator instance with four attributes. All attributes have their default values.
    // 
    // Arguments (Keywords):
    //     - sent   : for attribute sent_,   default False
    //     - alpha  : for attribute alpha_,  default 0.25
    //     - beta   : for attribute beta_,   default 0.10
    //     - output : for attribute output_, default sys.stdout
    // """
    RibesEvaluator(bool sent = false, double alpha = 0.25, double beta = 0.10, ostream & output = cout) :
        RIBES_VERSION_("1.02.3"), debug(0),
        sent_(sent), alpha_(alpha), beta_(beta), output_(&output)        
         { }

    string RIBES_VERSION_;
    int debug;
    bool sent_;
    double alpha_;
    double beta_;
    ostream * output_;

    // """Calculates Kendall's tau between a reference and a hypothesis
    //
    // Calculates Kendall's tau (also unigram precision and brevity penalty (BP))
    // between a reference word list and a system output (hypothesis) word list.
    //
    // Arguments:
    //     ref : list of reference words
    //     sub : list of system output (hypothesis) words
    //
    // Returns:
    //     A tuple (nkt, precision, bp)
    //         - nkt       : normalized Kendall's tau
    //         - precision : unigram precision
    //         - bp        : brevity penalty
    //
    // Raises:
    //     RuntimeError: reference has no words, possibly due to a format violation
    // """
    TauScore kendall(vector<string> ref, vector<string> hyp) {
        // cerr << "ref:"; BOOST_FOREACH(string i, ref) cerr << " " << i; cerr << endl;
        // cerr << "hyp:"; BOOST_FOREACH(string i, hyp) cerr << " " << i; cerr << endl;
    
        // check reference length, raise RuntimeError if no words are found.
        if(ref.size() == 0)
            throw runtime_error("Reference has no words");

        // check hypothesis length, return "zeros" if no words are found
        if(hyp.size() == 0)
            return TauScore(0.0, 0.0, 0.0);
    
        // calculate brevity penalty (BP), not exceeding 1.0
        double bp = min(1.0, exp(1.0 - 1.0 * ref.size()/hyp.size())); 
        
        // determine which ref. word corresponds to each hypothesis word
        // list for ref. word indices
        vector<int> intlist;
    
        // Find the positions of each word in each of the sentences
        unordered_map<string, vector<int> > ref_count, hyp_count;
        for(int i = 0; i < (int)ref.size(); i++)
            ref_count[ref[i]].push_back(i);
        for(int i = 0; i < (int)hyp.size(); i++)
            hyp_count[hyp[i]].push_back(i);
    
        for(int i = 0; i < (int)hyp.size(); i++) {
            // If hyp[i] doesn't exist in the reference, go to the next word
            if(ref_count.find(hyp[i]) == ref_count.end())
                continue;
            // Get matched words
            const vector<int> & ref_match = ref_count[hyp[i]];
            const vector<int> & hyp_match = hyp_count[hyp[i]];

            // if we can determine one-to-one word correspondence by only unigram
            // one-to-one correspondence
            if (ref_match.size() == 1 && hyp_match.size() == 1) {
                intlist.push_back(ref_match[0]);
            // if not, we consider context words
            } else {
                // These vectors store all hypotheses that are still matching on the right or left
                vector<int> left_ref = ref_match, left_hyp = hyp_match,
                            right_ref = ref_match, right_hyp = hyp_match;
                for(int window = 1; window < max(i, (int)hyp.size()-i); window++) {
                    // Update the possible hypotheses on the left
                    if(window <= i) {
                        vector<int> new_left_ref, new_left_hyp;
                        BOOST_FOREACH(int j, left_ref)
                            if(window <= j && ref[j-window] == hyp[i-window])
                                new_left_ref.push_back(j);
                        BOOST_FOREACH(int j, left_hyp)
                            if(window <= j && hyp[j-window] == hyp[i-window])
                                new_left_hyp.push_back(j);
                        if(new_left_ref.size() == 1 && new_left_hyp.size() == 1) {
                            intlist.push_back(new_left_ref[0]);
                            break;
                        }
                        left_ref = new_left_ref; left_hyp = new_left_hyp;
                    }
                    // Update the possible hypotheses on the right
                    if(i+window < (int)hyp.size()) {
                        vector<int> new_right_ref, new_right_hyp;
                        BOOST_FOREACH(int j, right_ref)
                            if(j+window < (int)ref.size() && ref[j+window] == hyp[i+window])
                                new_right_ref.push_back(j);
                        BOOST_FOREACH(int j, right_hyp)
                            if(j+window < (int)hyp.size() && hyp[j+window] == hyp[i+window])
                                new_right_hyp.push_back(j);
                        if(new_right_ref.size() == 1 && new_right_hyp.size() == 1) {
                            intlist.push_back(new_right_ref[0]);
                            break;
                        }
                        right_ref = new_right_ref; right_hyp = new_right_hyp;
                    }
                }
            }
        }
        // cerr << "intlist:"; BOOST_FOREACH(int i, intlist) cerr << " " << i; cerr << endl;
    
        // At least two word correspondences are needed for rank correlation
        int n = intlist.size();
        if (n == 1 && ref.size() == 1)
            return TauScore(1.0, 1.0/hyp.size(), bp);
        // if not, return score 0.0
        else if(n < 2)
            return TauScore(0.0, 0.0, bp);
    
        // calculation of rank correlation coefficient
        // count "ascending pairs" (intlist[i] < intlist[j])
        int ascending = 0;
        for(int i = 0; i < (int)intlist.size()-1; i++)
            for(int j = i+1; j < (int)intlist.size(); j++)
                if(intlist[i] < intlist[j])
                    ascending++;
    
        // normalize Kendall's tau
        double nkt = double(ascending) / ((n * (n - 1))/2);
    
        // calculate unigram precision
        double precision = 1.0 * n / hyp.size();
    
        // return tuple (Normalized Kendall's tau, Unigram Precision, and Brevity Penalty)
        // cerr << "nkt=" <<nkt << " precision=" << precision << " bp=" <<bp << endl;
        return TauScore(nkt, precision, bp);
    }
    

    // """Evaluate a system output with multiple references.
    // 
    // Calculates RIBES for a system output (hypothesis) with multiple references,
    // and returns "best" score among multi-references and individual scores.
    // The scores are corpus-wise, i.e., averaged by the number of sentences.
    // 
    // Arguments:
    //     hyp  : "Corpus" instance of hypothesis
    //     REFS : list of "Corpus" instances of references
    // 
    // Returns:
    //     A tuple (_best_ribes_acc, _RIBES_ACC)
    //         - _best_ribes_acc : best corpus-wise RIBES among multi-reference
    //         - _RIBES_ACC      : list of corpus-wise RIBES for each reference
    // 
    // Raises:
    //     RuntimeError : #sentences of hypothesis and reference doesn't match
    //     RuntimeError : from the function "kendall"
    // """
    pair<double, vector<double> > eval(const Corpus & hyp, const vector<Corpus> & REFS) { 
        // check #sentences of hypothesis and each of the multi-references
        BOOST_FOREACH(const Corpus & ref, REFS)
            if(hyp.size() != ref.size())
                throw runtime_error((format("Different #sentences between %1 (%2 sents.) and %3 (%4 sents.)")
                                                % hyp.filename() % hyp.size() % ref.filename() % ref.size()).str());
        
        // initialize "best" corpus-wise score
        double _best_ribes_acc = 0.0;
        // initialize individual corpus-wise score list
        vector<double> _RIBES_ACC(REFS.size(), 0.0);
    
        // scores each hypothesis
        for(int i = 0; i < (int)hyp.size(); i++) {
            // initialize "best" sentence-wise score
            double _best_ribes = 0.0;
    
            // for each reference
            for(int r = 0; r < (int)REFS.size(); r++) {
                // Calculate the score
                TauScore score = kendall(REFS[r][i], hyp[i]);
    
                // RIBES = (normalized Kendall's tau) * (unigram_precision ** alpha) * (brevity_penalty ** beta)
                double _ribes = score.nkt * (pow(score.precision, alpha_)) * (pow(score.bp, beta_));
                // accumulate RIBES for "individual" corpus-wise score for (r+1)-th reference
                _RIBES_ACC[r] += _ribes / hyp.size();
                // maintain the best sentence-wise score
                if(_ribes > _best_ribes)
                    _best_ribes = _ribes;
            }
    
            // accumulate the "best" sentence-wise score for the "best" corpus-wise score
            _best_ribes_acc += _best_ribes / hyp.size();
    
            // print "best" sentence-wise score if sent_ is True
            if (sent_ && output_ != NULL)
                *output_ << format("%1$.6f alpha=%2$f beta=%3$f %4$s sentence %5$d") % _best_ribes % alpha_ % beta_ % hyp.filename() % i << endl;
        }
    
        // returns the tuple of the "best" corpus-wise RIBES and score list for each reference
        return pair<double, vector<double> >(_best_ribes_acc, _RIBES_ACC);
    }
    
    // wrapper function for output
    static void outputRibes(const po::variables_map & options, ostream & out = cout) {

        // Check to make sure we have a reference
        if(!options.count("ref"))
            throw runtime_error("Must specify at least one reference with --ref or -r");
        if(!options.count("hyp"))
            throw runtime_error("Must specify at least one system output");

        // print start time
        time_t now = time(0);
        tm* localtm = localtime(&now);
        cerr << "# RIBES evaluation start at " << asctime(localtm);
    
        // initialize "RIBESevaluator" instance
        double alpha = options["alpha"].as<double>();
        double beta = options["beta"].as<double>();
        RibesEvaluator evaluator(options["sentence"].as<bool>(),alpha,beta,out);
        int debug = options["debug"].as<int>();
    
        // REFS : list of "Corpus" instance (for multi reference)
        vector<Corpus> REFS;
        BOOST_FOREACH(const string & ref, options["ref"].as< vector<string> >()) {
            // print reference file name (if debug > 0)
            if (debug > 0)
                out << "# reference file [" << REFS.size() << "] : " << ref << endl;
    
            // read multi references, construct and store "Corpus" instance
            REFS.push_back( Corpus(ref, options["case"].as<bool>()) );
        }
    
        const vector<string> & args = options["hyp"].as< vector<string> >();
        for(int i = 0; i < (int)args.size(); i++) {
            // print system output file name (if debug > 0)
            if(debug > 0) 
                out << "# system output file [" << i << "] : " << args[i] << endl;
    
            // read system output and construct "Corpus" instance
            Corpus hyp(args[i], options["case"].as<bool>());
    
            // evaluate by RIBES -- "best_ribes" stands for the best score by multi-references, RIBESs stands for the score list for each references
            pair<double, vector<double> > result = evaluator.eval(hyp, REFS);
    
            // print results
            out << format("%1$.6f alpha=%2$f beta=%3$f %4$s") % result.first % alpha % beta % args[i] << endl;
        }
    
        // print start time
        now = time(0);
        localtm = localtime(&now);
        cerr << "# RIBES evaluation done at " << asctime(localtm);
    }

};

// main function
int main(int argc, const char** argv) {

    string usage = "ribes-c [options] system_outputs";

    // option definitions
    po::options_description optparser(usage);
    optparser.add_options()
        // -d/--debug : debug level (0: scores and start/end time, 1: +ref/hyp files)
        ("debug,d", po::value<int>()->default_value(0), "debug level")
        // -r/--ref : reference (multiple references available, repeat "-r REF" in arguments)
        ("ref,r", po::value< vector<string> >(), "reference translation file (use multiple \"-r REF\" for multi-references)")
        // -h/--hyp : hypothesis files (multiple hypotheses available, repeat "-r REF" in arguments)
        ("hyp,h", po::value< vector<string> >(), "translation hypothesis file file (use multiple \"-h HYP\" to grade multiple files)")
        // -c/--case : preserve uppercase letters
        ("case,c", po::value< bool >()->default_value(false), "preserve uppercase letters in evaluation (default: False -- lowercasing all words)")
        // -s/--sentence : show scores for every sentences
        ("sentence,s", po::value< bool >()->default_value(false), "output scores for every sentences")
        // -a/--alpha : "Unigram Precison" to the {alpha}-th power
        ("alpha,a", po::value< double >()->default_value(0.25), "hyperparameter alpha (default=0.25)")
        // -b/--beta : "Brevity Penalty" to the {beta}-th power
        ("beta,b", po::value< double >()->default_value(0.10), "hyperparameter beta  (default=0.10)")
        // -o/--output : output file
        ("output,o", po::value< string >()->default_value(""), "log output file")
    ;
    po::positional_options_description args;
    args.add("hyp", -1);

    // parse options
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(optparser).positional(args).run(), vm);
    po::notify(vm);

    // Run Ribes and output
    string output =  vm["output"].as<string>();
    if(output.length() == 0) {
        RibesEvaluator::outputRibes(vm);
    } else {
        ofstream out(output.c_str());
        RibesEvaluator::outputRibes(vm, out);
    }

    return 0;
}
