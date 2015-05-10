#include "translator.h"

void read_config(Filenames &fns,Parameter &para, Weight &weight, const string &config_file)
{
	ifstream fin;
	fin.open(config_file.c_str());
	if (!fin.is_open())
	{
		cerr<<"fail to open config file\n";
		return;
	}
	string line;
	while(getline(fin,line))
	{
		TrimLine(line);
		if (line == "[input-file]")
		{
			getline(fin,line);
			fns.input_file = line;
		}
		else if (line == "[output-file]")
		{
			getline(fin,line);
			fns.output_file = line;
		}
		else if (line == "[nbest-file]")
		{
			getline(fin,line);
			fns.nbest_file = line;
		}
		else if (line == "[src-vocab-file]")
		{
			getline(fin,line);
			fns.src_vocab_file = line;
		}
		else if (line == "[tgt-vocab-file]")
		{
			getline(fin,line);
			fns.tgt_vocab_file = line;
		}
		else if (line == "[rule-table-file]")
		{
			getline(fin,line);
			fns.rule_table_file = line;
		}
		else if (line == "[lm-file]")
		{
			getline(fin,line);
			fns.lm_file = line;
		}
		else if (line == "[function-words-file]")
		{
			getline(fin,line);
			fns.fw_file = line;
		}
		else if (line == "[BEAM-SIZE]")
		{
			getline(fin,line);
			para.BEAM_SIZE = stoi(line);
		}
		else if (line == "[CUBE-SIZE]")
		{
			getline(fin,line);
			para.CUBE_SIZE = stoi(line);
		}
		else if (line == "[SEN-THREAD-NUM]")
		{
			getline(fin,line);
			para.SEN_THREAD_NUM = stoi(line);
		}
		else if (line == "[SPAN-THREAD-NUM]")
		{
			getline(fin,line);
			para.SPAN_THREAD_NUM = stoi(line);
		}
		else if (line == "[NBEST-NUM]")
		{
			getline(fin,line);
			para.NBEST_NUM = stoi(line);
		}
		else if (line == "[RULE-NUM-LIMIT]")
		{
			getline(fin,line);
			para.RULE_NUM_LIMIT = stoi(line);
		}
		else if (line == "[PRINT-NBEST]")
		{
			getline(fin,line);
			para.PRINT_NBEST = stoi(line);
		}
		else if (line == "[DUMP-RULE]")
		{
			getline(fin,line);
			para.DUMP_RULE = stoi(line);
		}
		else if (line == "[DROP-OOV]")
		{
			getline(fin,line);
			para.DROP_OOV = stoi(line);
		}
		else if (line == "[weight]")
		{
			while(getline(fin,line))
			{
				if (line == "")
					continue;
				stringstream ss(line);
				string feature;
				ss >> feature;
				if (feature.find("trans") != string::npos)
				{
					double w;
					ss>>w;
					weight.trans.push_back(w);
				}
				else if(feature.find("len") != string::npos)
				{
					ss>>weight.len;
				}
				else if(feature.find("lm") != string::npos)
				{
					ss>>weight.lm;
				}
				else if(feature.find("rule") != string::npos)
				{
					ss>>weight.rule_num;
				}
				else if(feature.find("glue") != string::npos)
				{
					ss>>weight.glue;
				}
			}
		}
	}
}

void parse_args(int argc, char *argv[],Filenames &fns,Parameter &para, Weight &weight)
{
	if (argc == 1)
	{
		read_config(fns,para,weight,"config.ini");
	}
	for( int i=1; i<argc; i++ )
	{
		string arg( argv[i] );
		if( arg == "-config" )
		{
			read_config(fns,para,weight,argv[++i]);
		}
		else if( arg == "-n-best-list" )
		{
			fns.nbest_file = argv[++i];
			para.NBEST_NUM = stoi(argv[++i]);
		}
		else if( arg == "-weight-overwrite" )
		{
			string weight_str = argv[++i];
			vector<string> vs;
			Split(vs,weight_str);
			for (size_t j=0; j<vs.size(); j++)
			{
				if (vs[j].find("transa") != string::npos)
				{
					weight.trans[0] = stod(vs[++j]);
				}
				else if (vs[j].find("transb") != string::npos)
				{
					weight.trans[1] = stod(vs[++j]);
				}
				else if (vs[j].find("transc") != string::npos)
				{
					weight.trans[2] = stod(vs[++j]);
				}
				else if (vs[j].find("transd") != string::npos)
				{
					weight.trans[3] = stod(vs[++j]);
				}
				else if (vs[j].find("lm") != string::npos)
				{
					weight.lm = stod(vs[++j]);
				}
				else if (vs[j].find("len") != string::npos)
				{
					weight.len = stod(vs[++j]);
				}
				else if (vs[j].find("rule") != string::npos)
				{
					weight.rule_num = stod(vs[++j]);
				}
				else if (vs[j].find("glue") != string::npos)
				{
					weight.glue = stod(vs[++j]);
				}
			}
		}
		else if( arg == "-show-weights" )
		{
			for (size_t j=0; j<weight.trans.size(); j++)
			{
				cout<<"trans"<<(char)('a'+j)<<"0= "<<weight.trans[j]<<endl;
			}
			cout<<"lm0= "<<weight.lm<<endl;
			cout<<"len0= "<<weight.len<<endl;
			cout<<"rule0= "<<weight.rule_num<<endl;
			cout<<"glue0= "<<weight.glue<<endl;
			exit(0);
		}
	}
}

void translate_file(const Models &models, const Parameter &para, const Weight &weight, const Filenames &fns)
{
	ifstream fin(fns.input_file.c_str());
	if (!fin.is_open())
	{
		cerr<<"cannot open input file!\n";
		return;
	}
	ofstream fout(fns.output_file.c_str());
	if (!fout.is_open())
	{
		cerr<<"cannot open output file!\n";
		return;
	}
	vector<string> input_sen;
	vector<string> output_sen;
	vector<vector<TuneInfo> > nbest_tune_info_list;
	vector<vector<string> > applied_rules_list;
	string line;
	while(getline(fin,line))
	{
		TrimLine(line);
		input_sen.push_back(line);
		vector<string> vs;
		Split(vs,line);
		for (const auto &word : vs)
		{
			models.src_vocab->get_id(word);                                             //避免并行时同时修改vocab发生冲突
		}
	}
	int sen_num = input_sen.size();
	output_sen.resize(sen_num);
	nbest_tune_info_list.resize(sen_num);
	applied_rules_list.resize(sen_num);
#pragma omp parallel for num_threads(para.SEN_THREAD_NUM)
	for (size_t i=0;i<sen_num;i++)
	{
		SentenceTranslator sen_translator(models,para,weight,input_sen.at(i));
		output_sen.at(i) = sen_translator.translate_sentence();
		if (para.PRINT_NBEST == true)
		{
			nbest_tune_info_list.at(i) = sen_translator.get_tune_info(i);
		}
		if (para.DUMP_RULE == true)
		{
			applied_rules_list.at(i) = sen_translator.get_applied_rules(i);
		}
	}
	for (const auto &sen : output_sen)
	{
		fout<<sen<<endl;
		cout<<sen<<endl;
	}
	if (para.PRINT_NBEST == true)
	{
		ofstream fnbest(fns.nbest_file);
		if (!fnbest.is_open())
		{
			cerr<<"cannot open nbest file!\n";
			return;
		}
		for (const auto &nbest_tune_info : nbest_tune_info_list)
		{
			for (const auto &tune_info : nbest_tune_info)
			{
				fnbest<<tune_info.sen_id<<" ||| "<<tune_info.translation<<" ||| ";
				for (size_t i=0; i<PROB_NUM; i++)
				{
					fnbest<<"trans"<<(char)('a'+i)<<"0= "<<tune_info.feature_values[i]<<" ";
				}
				fnbest<<"lm0= "<<tune_info.feature_values[4]<<" ";
				fnbest<<"len0= "<<tune_info.feature_values[5]<<" ";
				fnbest<<"rule0= "<<tune_info.feature_values[6]<<" ";
				fnbest<<"glue0= "<<tune_info.feature_values[7]<<" ";
				fnbest<<"||| "<<tune_info.total_score<<endl;
			}
		}
	}
	if (para.DUMP_RULE == true)
	{
		ofstream frules("applied-rules.txt");
		if (!frules.is_open())
		{
			cerr<<"cannot open applied-rules file!\n";
			return;
		}
		size_t n=0;
		for (const auto &applied_rules : applied_rules_list)
		{
			//frules<<++n<<endl;
			for (const auto &applied_rule : applied_rules)
			{
				//frules<<applied_rule<<endl;
				frules<<applied_rule;
			}
			frules<<endl;
		}
	}
}

void load_function_words(set<int> &src_function_words,const string &function_words_file,Vocab *src_vocab)
{
	ifstream fin(function_words_file.c_str());
	if (!fin.is_open())
	{
		cerr<<"cannot open function words file!\n";
		return;
	}
	string line;
	while(getline(fin,line))
	{
		TrimLine(line);
		src_function_words.insert(src_vocab->get_id(line));
	}
}

int main( int argc, char *argv[])
{
	clock_t a,b;
	a = clock();

	omp_set_nested(1);
	Filenames fns;
	Parameter para;
	Weight weight;
	parse_args(argc,argv,fns,para,weight);

	Vocab *src_vocab = new Vocab(fns.src_vocab_file);
	Vocab *tgt_vocab = new Vocab(fns.tgt_vocab_file);
	RuleTable *ruletable = new RuleTable(para.RULE_NUM_LIMIT,weight,fns.rule_table_file);
	LanguageModel *lm_model = new LanguageModel(fns.lm_file,tgt_vocab);
	set<int> src_function_words;
	load_function_words(src_function_words,fns.fw_file,src_vocab);

	b = clock();
	cerr<<"loading time: "<<double(b-a)/CLOCKS_PER_SEC<<endl;

	Models models = {src_vocab,tgt_vocab,ruletable,lm_model,&src_function_words};
	translate_file(models,para,weight,fns);
	b = clock();
	cerr<<"time cost: "<<double(b-a)/CLOCKS_PER_SEC<<endl;
	return 0;
}
