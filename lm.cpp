#include "lm.h"

struct ID_converter : public lm::EnumerateVocab 
{
	ID_converter(vector<lm::WordIndex>* out, Vocab* vocab) : sub_to_kenlm_id(out), UNK_ID(0),tgt_vocab(vocab) { sub_to_kenlm_id->clear(); }
	void Add(lm::WordIndex index, const StringPiece &str) 
	{
		const int ori_id = tgt_vocab->get_id(str.as_string());
		if (ori_id >= sub_to_kenlm_id->size())
		{
			sub_to_kenlm_id->resize(ori_id + 1, UNK_ID);
		}
		sub_to_kenlm_id->at(ori_id) = index;
	}
	vector<lm::WordIndex>* sub_to_kenlm_id;
	const lm::WordIndex UNK_ID;
	Vocab* tgt_vocab;
};

LanguageModel::LanguageModel(const string &lm_file, Vocab *tgt_vocab)
{
	ID_converter id_converter(&ori_to_kenlm_id,tgt_vocab);
	Config conf;
	conf.enumerate_vocab = &id_converter;
	kenlm = new Model(lm_file.c_str(), conf);
	EOS = convert_to_kenlm_id(tgt_vocab->get_id("</s>"));
	nonterminal_wid = tgt_vocab->get_id("[X][X]");
	cout<<"load language model file "<<lm_file<<" over\n";
};

lm::WordIndex LanguageModel::convert_to_kenlm_id(int wid)
{
	if (wid >= ori_to_kenlm_id.size())
		return 0;
	else
		return ori_to_kenlm_id[wid];
}

double LanguageModel::cal_increased_lm_score(Cand* cand) 
{
	RuleScore<Model> rule_score(*kenlm,cand->lm_state);
	if (cand->applied_rule.tgt_rule == NULL)            //OOV候选
	{
		const lm::WordIndex ken_lm_id = convert_to_kenlm_id(cand->tgt_wids.at(0));
		rule_score.Terminal(ken_lm_id);
	}
	else
	{
		int nt_num = 1;
		for (auto wid : cand->applied_rule.tgt_rule->wids)
		{
			if (wid == nonterminal_wid)
			{
				if (nt_num == 1)
				{
					rule_score.NonTerminal((cand->child_x1)->lm_state);
					nt_num += 1;
				}
				else
				{
					rule_score.NonTerminal((cand->child_x2)->lm_state);
				}
			}
			else
			{
				const lm::WordIndex ken_lm_id = convert_to_kenlm_id(wid);
				rule_score.Terminal(ken_lm_id);
			}
		}
	}
	double increased_lm_score = rule_score.Finish();
	cand->lm_state.ZeroRemaining();
	return increased_lm_score;
}

double LanguageModel::cal_final_increased_lm_score(Cand* cand) 
{
	ChartState cstate;
	RuleScore<Model> rule_score(*kenlm, cstate);
	rule_score.BeginSentence();
	rule_score.NonTerminal(cand->lm_state, 0.0f);
	rule_score.Terminal(EOS);
	return rule_score.Finish();
}
