#include "ruletable.h"

void RuleTable::load_rule_table(const string &rule_table_file)
{
	ifstream fin(rule_table_file.c_str(),ios::binary);
	if (!fin.is_open())
	{
		cerr<<"cannot open rule table file!\n";
		return;
	}
	short int src_rule_len=0;
	while(fin.read((char*)&src_rule_len,sizeof(short int)))
	{
		vector<int> src_wids;
		src_wids.resize(src_rule_len);
		fin.read((char*)&src_wids[0],sizeof(int)*src_rule_len);

		short int tgt_rule_len=0;
		fin.read((char*)&tgt_rule_len,sizeof(short int));
		if (tgt_rule_len > RULE_LEN_MAX)
		{
			cout<<"error, rule length exceed, bye\n";
			exit(EXIT_FAILURE);
		}
		TgtRule tgt_rule;
		tgt_rule.word_num = tgt_rule_len;
		tgt_rule.wids.resize(tgt_rule_len);
		fin.read((char*)&(tgt_rule.wids[0]),sizeof(int)*tgt_rule_len);

		tgt_rule.probs.resize(PROB_NUM);
		fin.read((char*)&(tgt_rule.probs[0]),sizeof(double)*PROB_NUM);

		tgt_rule.score = 0;
		if( tgt_rule.probs.size() != weight.trans.size() )
		{
			cout<<"number of probability in rule is wrong!"<<endl;
		}
		for( size_t i=0; i<weight.trans.size(); i++ )
		{
			tgt_rule.score += tgt_rule.probs[i]*weight.trans[i];
		}

		short int rule_type;
		fin.read((char*)&rule_type,sizeof(short int));
		tgt_rule.rule_type = rule_type;
		add_rule_to_trie(src_wids,tgt_rule);
	}
	fin.close();
	cout<<"load rule table file "<<rule_table_file<<" over\n";
}

vector<vector<TgtRule>* > RuleTable::find_matched_rules_for_prefixes(const vector<int> &src_wids,const size_t pos)
{
	vector<vector<TgtRule>* > matched_rules_for_prefixes;
	RuleTrieNode* current = root;
	for (size_t i=pos;i<src_wids.size() && i-pos<RULE_LEN_MAX;i++)
	{
		auto it = current->id2subtrie_map.find(src_wids.at(i));
		if (it != current->id2subtrie_map.end())
		{
			current = it->second;
			if (current->tgt_rules.size() == 0)
			{
				matched_rules_for_prefixes.push_back(NULL);
			}
			else
			{
				matched_rules_for_prefixes.push_back(&(current->tgt_rules));
			}
		}
		else
		{
			matched_rules_for_prefixes.push_back(NULL);
			return matched_rules_for_prefixes;
		}
	}
	return matched_rules_for_prefixes;
}

void RuleTable::add_rule_to_trie(const vector<int> &src_wids, const TgtRule &tgt_rule)
{
	RuleTrieNode* current = root;
	for (const auto &wid : src_wids)
	{        
		auto it = current->id2subtrie_map.find(wid);
		if ( it != current->id2subtrie_map.end() )
		{
			current = it->second;
		}
		else
		{
			RuleTrieNode* tmp = new RuleTrieNode();
			current->id2subtrie_map.insert(make_pair(wid,tmp));
			current = tmp;
		}
	}
	if (current->tgt_rules.size() < RULE_NUM_LIMIT)
	{
		current->tgt_rules.push_back(tgt_rule);
	}
	else
	{
		auto it = min_element(current->tgt_rules.begin(), current->tgt_rules.end());
		if( it->score < tgt_rule.score )
		{
			(*it) = tgt_rule;
		}
	}
}
