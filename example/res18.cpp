#include"net.h"
#include"benchmark.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"




static void normize(const easynn::Mat& m,std::vector<float> mean,std::vector<float> var)
{
    for (int q=0; q<m.c; q++)
    {
        float* ptr = m.channel(q);
        for (int z=0; z<m.d; z++)
        {
            for (int y=0; y<m.h; y++)
            {
                for (int x=0; x<m.w; x++)
                {
                    ptr[x] = (ptr[x]/255-mean[q])/var[q];
                    
                }
                ptr += m.w; 
            }
        }
    }
}

static int findMax(const easynn::Mat& m)
{
    int index=0;
    float max = m[0];
    for (int x=0; x<m.w; x++)
    {
        if(m[x]>max)
        {
            index=x;
            max = m[x];
        }
    }
    printf("max:%f\n",max);
    return index;
}

static void pretreatment(cv::Mat& input_image,easynn::Mat& output_image,int h,int w)
{
    cv::Mat resize_image;
    cv::resize(input_image, resize_image, cv::Size(224, 224));

    cv::Mat rgb_image;
    cv::cvtColor(resize_image, rgb_image, cv::COLOR_BGR2RGB);
    rgb_image.convertTo(rgb_image, CV_32FC3);
    std::vector<cv::Mat> split_images;
    cv::split(rgb_image, split_images);

    
    output_image.create(w,h,3);

    int index = 0;
    for (const auto& split_image : split_images)
    {
        memcpy((void*)output_image.channel(index), split_image.data, sizeof(float) * split_image.total());
        index += 1;
    }
} 

static void res18(const easynn::Mat& in,easynn::Mat& result)
{
    easynn::Net net;
    net.loadModel(\
    "../example/res18.pnnx.param",\
    "../example/res18.pnnx.bin");
    net.input(0,in);
    net.extractBlob(49,result);
}

int main()
{
    std::string image_path = "../images/dog.jpg";
    cv::Mat image = cv::imread(image_path, 1);
    if (image.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", image_path.c_str());
        return -1;
    }

    //cv::Mat to EasyNN Mat
    easynn::Mat in;
    pretreatment(image,in,224,224);

    //normize 
    std::vector<float> mean = {0.485f,0.456f,0.406f};
    std::vector<float> var = { 0.229f,0.224f,0.225f};
    normize(in,mean,var);

    // forward net
    easynn::Mat result;
    double start = easynn::get_current_time();
    res18(in,result);
    double end = easynn::get_current_time();

    printf("total time is %f ms\n",end-start);
    
    
    //find Max score class
    int cls = findMax(result);
    printf("cls = %d\n",cls);

    return 0;
}