#include "stdafx.h"
#include "cand.h"
#include "vocab.h"
//#include "ruletable.h"
#include "lm.h"
#include "myutils.h"

struct Models
{
	Vocab *src_vocab;
	Vocab *tgt_vocab;
	RuleTable *ruletable;
	LanguageModel *lm_model;
	set<int> *src_function_words;
};

struct SyntaxNode
{
	string label;                      	//该节点的句法标签或者词
	SyntaxNode* father;
	vector<SyntaxNode*> children;
	pair<int,int> span;                	//该节点的span
	
	SyntaxNode ()
	{
		father   = NULL;
		span = make_pair(-1,-1);
	}
	~SyntaxNode ()
	{
		for (auto node : children)
		{
			delete node;
		}
	}
};

class SentenceTranslator
{
	public:
		SentenceTranslator(const Models &i_models, const Parameter &i_para, const Weight &i_weight, const string &input_sen);
		~SentenceTranslator();
		string translate_sentence();
		vector<TuneInfo> get_tune_info(size_t sen_id);
		vector<string> get_applied_rules(size_t sen_id);
	private:
		void build_tree_from_str(const string &line_tree);
		void cal_span_for_each_node(SyntaxNode *node);
		void fill_span2cands_with_phrase_rules();
		void fill_span2rules_with_hiero_rules();
		void fill_span2rules_with_AX_XA_XAX_rule();
		void fill_span2rules_with_AXB_AXBX_XAXB_rule();
		void fill_span2rules_with_AXBXC_rule();
		void fill_span2rules_with_glue_rule();
		void fill_span2rules_with_matched_rules(vector<TgtRule> &matched_rules,vector<int> &src_ids,pair<int,int> span,pair<int,int> span_src_x1,pair<int,int> span_src_x2);
		void generate_kbest_for_span(const size_t beg,const size_t span);
		void generate_cand_with_rule_and_add_to_pq(Rule &rule,int rank_x1,int rank_x2,Candpq &new_cands_by_mergence);
		void add_neighbours_to_pq(Cand *cur_cand, Candpq &new_cands_by_mergence);
		void dump_rules(vector<string> &applied_rules, Cand *cand);
		string words_to_str(vector<int> wids, int drop_oov);
		bool is_only_function_words_in_span(pair<int,int> span_X);

	private:
		Vocab *src_vocab;
		Vocab *tgt_vocab;
		RuleTable *ruletable;
		LanguageModel *lm_model;
		set<int> *src_function_words;
		Parameter para;
		Weight feature_weight;

		vector<vector<CandBeam> > span2cands;		    //存储解码过程中所有跨度对应的候选列表, 
													    //span2cands[i][j]存储起始位置为i, 跨度为j的候选列表
		vector<vector<vector<Rule> > > span2rules;	    //存储每个跨度所有能用的hiero规则

		SyntaxNode *root;
		vector<int> src_wids;
		vector<int> verb_flags;
		vector<int> fw_flags;
		vector<vector<bool> > span_to_node_flag;
		size_t src_sen_len;
		int src_nt_id;                                  //源端非终结符的id
		int tgt_nt_id; 									//目标端非终结符的id
};
