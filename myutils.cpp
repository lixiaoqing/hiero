#include "myutils.h"

void Split(vector <string> &vs, string &s)
{
	vs.clear();
	stringstream ss;
	string e;
	ss << s;
	while(ss >> e)
		vs.push_back(e);
}

void Split(vector <string> &vs, string &s, string &sep)
{
	int cur = 0,next;
	next = s.find(sep);
	while(next != string::npos)
	{
		if(s.substr(cur,next-cur) !="")
			vs.push_back(s.substr(cur,next-cur));
		cur = next+sep.size();
		next = s.find(sep,cur);
	}
	vs.push_back(s.substr(cur));
}

void TrimLine(string &line)
{
	line.erase(0,line.find_first_not_of(" \t\r\n"));
	line.erase(line.find_last_not_of(" \t\r\n")+1);
}
