


#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>

void copyfile(const std::string &from, const std::string &to)
{
    boost::filesystem::path pfrom(from);
    boost::filesystem::path pto(to);

    boost::filesystem::copy_file(pfrom,pto);
}
