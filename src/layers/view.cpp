#include<iostream>
#include"view.h"


namespace easynn{

View::View()
{
    one_blob_only = true;
}
int View::forward(const Mat& input,Mat& output,const Optional& op)
{
    output = input.clone();
    std::cout<<"View forward"<<std::endl;
    return 0;
}

}//namespace