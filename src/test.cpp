
#include <iostream>

int main(int argc, char **argv)
{
	int N;
	int tmp;
	long sum=0;
    int first,last;

	std::cin >> N;
    std::cin >> first;
	
    for(int i=1; i<N-2; i++){
		std::cin >> tmp;
		sum += tmp;
	}

    std::cin >> last;

    long realsum = N*(first+last)/2;

    std::cout << realsum - sum - first - last<< std::endl;

}
