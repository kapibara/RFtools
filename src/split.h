#ifndef SPLIT_H

#define SPLIT_H

#include <string>
#include <vector>

void split(const std::string &str, const std::string &symbols, std::vector<std::string> &result);

std::string replace_substr(const std::string &input, const std::string &toreplace, const std::string &replacewith);

#endif
