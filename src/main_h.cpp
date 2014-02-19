#include <iostream>
#include <string>

class Handle{
public:
    Handle(){
        std::cout << "Handle()" << std::endl;
    }
    ~Handle(){
        std::cout << "~Handle()" << std::endl;
    }
    virtual std::string Name(){
        return "Handle";
    }
};

class MyHandle: public Handle{
public:
    MyHandle(){
        std::cout << "MyHandle()" << std::endl;
    }
    ~MyHandle(){
        std::cout << "~MyHandle()" << std::endl;
    }
    virtual std::string Name(){
        return "MyHandle";
    }
};

class TestBase
{
public:
    TestBase(Handle &h){
        std::cout << "TestBase()" << std::endl;
        doSomething(h);
    }
private:
    void doSomething(Handle &h){
        std::cout << h.Name() << std::endl;
    }
};

class Test: public TestBase
{
public:
    Test():TestBase(MyHandle()){
        std::cout << "Test()" << std::endl;
    }
    void printMe(){
        std::cout << "I am here" << std::endl;
    }
};

int main(int argc, char **argv)
{
    Test t;
    t.printMe();
    std::cout << "Test passed" << std::endl;
}
