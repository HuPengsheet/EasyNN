#include"test_ulti.h"
#include"test_fun.h"
#include"ir.h"
#include"net.h"
#include"mat.h"
#include"layers/cxx/adaptiveavgpool.h"

std::vector<std::vector<std::vector<float>>> ada_input_data=\
        {{{0.2295, 0.2599, 0.6441, 0.0168, 0.7927, 0.3168, 0.9468, 0.2946,
           0.6510, 0.1767, 0.7645, 0.4612},
          {0.3943, 0.9388, 0.9126, 0.3142, 0.2770, 0.9274, 0.2598, 0.9339,
           0.4860, 0.6099, 0.4828, 0.5948},
          {0.9998, 0.1657, 0.4309, 0.5119, 0.3204, 0.4685, 0.3708, 0.3458,
           0.6965, 0.7607, 0.4896, 0.3414},
          {0.8516, 0.7120, 0.7675, 0.5260, 0.1127, 0.4690, 0.0656, 0.1274,
           0.7237, 0.8612, 0.9578, 0.0717},
          {0.7778, 0.9389, 0.4477, 0.9347, 0.4932, 0.6393, 0.2195, 0.4559,
           0.1840, 0.9226, 0.1510, 0.1915},
          {0.3332, 0.2625, 0.9882, 0.6747, 0.0630, 0.0969, 0.1638, 0.2066,
           0.7487, 0.2935, 0.1886, 0.2975},
          {0.6795, 0.6518, 0.5196, 0.0989, 0.2921, 0.9160, 0.9370, 0.3772,
           0.0927, 0.2677, 0.7827, 0.2746},
          {0.0812, 0.1388, 0.5617, 0.6608, 0.4961, 0.8730, 0.1547, 0.4763,
           0.0420, 0.8068, 0.1394, 0.3010},
          {0.0321, 0.3149, 0.5720, 0.6835, 0.1845, 0.0031, 0.2095, 0.2987,
           0.9365, 0.2844, 0.7457, 0.8291},
          {0.3599, 0.6601, 0.8168, 0.4337, 0.8746, 0.5469, 0.3573, 0.4779,
           0.2486, 0.8504, 0.6616, 0.9320},
          {0.3530, 0.5634, 0.5081, 0.7176, 0.1062, 0.3808, 0.9981, 0.4353,
           0.3269, 0.6623, 0.3985, 0.7898},
          {0.9516, 0.0289, 0.1992, 0.1402, 0.2444, 0.1211, 0.1212, 0.0188,
           0.2831, 0.6390, 0.9189, 0.1517}},

         {{0.9418, 0.8566, 0.1098, 0.0454, 0.1247, 0.9565, 0.8543, 0.4707,
           0.3405, 0.7362, 0.9467, 0.7226},
          {0.6276, 0.2356, 0.0118, 0.8911, 0.4193, 0.6709, 0.9051, 0.1855,
           0.8339, 0.3113, 0.4981, 0.9342},
          {0.1486, 0.3907, 0.0704, 0.2226, 0.5163, 0.9326, 0.1216, 0.9232,
           0.8832, 0.3250, 0.1453, 0.5212},
          {0.7148, 0.5316, 0.3406, 0.0170, 0.3191, 0.2849, 0.2415, 0.5319,
           0.7618, 0.9995, 0.9593, 0.6969},
          {0.9408, 0.8048, 0.1872, 0.3158, 0.9612, 0.6753, 0.9354, 0.5379,
           0.3592, 0.4402, 0.2048, 0.0823},
          {0.7723, 0.6818, 0.7118, 0.6614, 0.7270, 0.5569, 0.6218, 0.3408,
           0.2093, 0.7379, 0.3721, 0.4468},
          {0.3118, 0.0977, 0.1137, 0.1801, 0.2204, 0.1414, 0.8682, 0.1054,
           0.8015, 0.2708, 0.6153, 0.6022},
          {0.8705, 0.5935, 0.9919, 0.0586, 0.9524, 0.9755, 0.1864, 0.5465,
           0.4812, 0.6421, 0.6969, 0.8026},
          {0.3568, 0.3298, 0.4970, 0.9867, 0.5367, 0.6762, 0.4278, 0.9811,
           0.1398, 0.8697, 0.5646, 0.3473},
          {0.8531, 0.9425, 0.6495, 0.4983, 0.8105, 0.4635, 0.8852, 0.0164,
           0.3397, 0.8758, 0.4604, 0.7227},
          {0.2770, 0.5340, 0.9153, 0.2417, 0.9824, 0.0425, 0.8894, 0.1229,
           0.9444, 0.2923, 0.2484, 0.8285},
          {0.1150, 0.5183, 0.5363, 0.5093, 0.0094, 0.2796, 0.5879, 0.4254,
           0.5275, 0.5013, 0.3831, 0.2962}},

         {{0.3096, 0.2493, 0.8229, 0.2754, 0.5033, 0.3588, 0.4517, 0.9865,
           0.6831, 0.1928, 0.3726, 0.9510},
          {0.5593, 0.9482, 0.7583, 0.5221, 0.0553, 0.6887, 0.1079, 0.8163,
           0.2664, 0.8677, 0.3635, 0.9620},
          {0.4066, 0.2021, 0.4684, 0.3672, 0.2404, 0.9130, 0.2811, 0.7541,
           0.5825, 0.5392, 0.1108, 0.4390},
          {0.2816, 0.7801, 0.3829, 0.0783, 0.3744, 0.2429, 0.8034, 0.6760,
           0.5320, 0.4850, 0.4305, 0.3114},
          {0.0435, 0.5865, 0.8042, 0.4533, 0.1508, 0.4364, 0.3380, 0.1109,
           0.7376, 0.0349, 0.4138, 0.2180},
          {0.2032, 0.2016, 0.6599, 0.8047, 0.4858, 0.3934, 0.9137, 0.2693,
           0.6279, 0.4159, 0.8641, 0.1526},
          {0.5429, 0.8051, 0.5889, 0.2773, 0.9594, 0.0128, 0.0453, 0.8524,
           0.8258, 0.7511, 0.1315, 0.4201},
          {0.2756, 0.8755, 0.7852, 0.8273, 0.8024, 0.9469, 0.0486, 0.9950,
           0.9743, 0.9827, 0.4943, 0.0917},
          {0.9856, 0.0872, 0.9139, 0.1970, 0.2668, 0.8008, 0.8758, 0.1363,
           0.2088, 0.5141, 0.5744, 0.0157},
          {0.1499, 0.4181, 0.6614, 0.1974, 0.6969, 0.3377, 0.8968, 0.6852,
           0.2181, 0.8499, 0.8425, 0.5653},
          {0.6385, 0.9671, 0.6611, 0.5936, 0.0432, 0.7459, 0.0015, 0.7680,
           0.4933, 0.8672, 0.3020, 0.1445},
          {0.0239, 0.0835, 0.6143, 0.0222, 0.7029, 0.3813, 0.3447, 0.4178,
           0.1323, 0.3780, 0.7844, 0.0424}}};

std::vector<std::vector<std::vector<float>>> ada_out_data1 = {{{0.4757}},{{0.5269}},{{0.4905}}};

std::vector<std::vector<std::vector<float>>> ada_out_data2=\
        {{{0.5281, 0.4583},
        {0.4379, 0.4785}},

        {{0.5105, 0.5594},
        {0.5019, 0.5360}},

        {{0.4448, 0.5018},
        {0.5247, 0.4909}}};

std::vector<std::vector<std::vector<float>>> ada_out_data3=\
        {{{0.5150, 0.5198},
         {0.5258, 0.3615},
         {0.4082, 0.5240}},

        {{0.4325, 0.6187},
         {0.5627, 0.4962},
         {0.5234, 0.5282}},

        {{0.4495, 0.5403},
         {0.5384, 0.4879},
         {0.4663, 0.4608}}};

std::vector<std::vector<std::vector<float>>> ada_out_data4 = \
        {{{0.4556, 0.6888, 0.3502, 0.5785, 0.6088, 0.5914, 0.5085, 0.5758},
         {0.6247, 0.6120, 0.3559, 0.4983, 0.4776, 0.6155, 0.5857, 0.4771},
         {0.6823, 0.5190, 0.3677, 0.3426, 0.2274, 0.4733, 0.7673, 0.4651},
         {0.8201, 0.7165, 0.5167, 0.4286, 0.2171, 0.3727, 0.7232, 0.3430},
         {0.5781, 0.6593, 0.5414, 0.3231, 0.2614, 0.3988, 0.3889, 0.2071},
         {0.4818, 0.6055, 0.2822, 0.3420, 0.4211, 0.3563, 0.3831, 0.3859},
         {0.3878, 0.4680, 0.3870, 0.6443, 0.4863, 0.2471, 0.4992, 0.3744},
         {0.1418, 0.3969, 0.5062, 0.3892, 0.2848, 0.4384, 0.4941, 0.5038},
         {0.3417, 0.5910, 0.5441, 0.4023, 0.3359, 0.4904, 0.6355, 0.7921},
         {0.4841, 0.6371, 0.5330, 0.4771, 0.5671, 0.3722, 0.6432, 0.6955},
         {0.4742, 0.3249, 0.3021, 0.2131, 0.3933, 0.2660, 0.6547, 0.5647}},

        {{0.6654, 0.3034, 0.3701, 0.5429, 0.6039, 0.4577, 0.6231, 0.7754},
         {0.3506, 0.1771, 0.5123, 0.6348, 0.5339, 0.7064, 0.3199, 0.5247},
         {0.4464, 0.3333, 0.2688, 0.5132, 0.4546, 0.7750, 0.6073, 0.5807},
         {0.7480, 0.4660, 0.4033, 0.5601, 0.5617, 0.5477, 0.6509, 0.4858},
         {0.7999, 0.5964, 0.6664, 0.7301, 0.6090, 0.3618, 0.4387, 0.2765},
         {0.4659, 0.4013, 0.4472, 0.4114, 0.4841, 0.3643, 0.4990, 0.5091},
         {0.4684, 0.4492, 0.3529, 0.5724, 0.4266, 0.4837, 0.5563, 0.6793},
         {0.5376, 0.6031, 0.6336, 0.7852, 0.5355, 0.5372, 0.6933, 0.6029},
         {0.6206, 0.6047, 0.7081, 0.6217, 0.5776, 0.3692, 0.6926, 0.5237},
         {0.6516, 0.7603, 0.6332, 0.5747, 0.4785, 0.3559, 0.4692, 0.5650},
         {0.3611, 0.6260, 0.4357, 0.3285, 0.5064, 0.5051, 0.3563, 0.4391}},

        {{0.5166, 0.6947, 0.3390, 0.4015, 0.5906, 0.6881, 0.4491, 0.6623},
         {0.5290, 0.5943, 0.2962, 0.4744, 0.4898, 0.6048, 0.4703, 0.4688},
         {0.4176, 0.4584, 0.2651, 0.4427, 0.6286, 0.6362, 0.3914, 0.3229},
         {0.4229, 0.6384, 0.2642, 0.3011, 0.4821, 0.5141, 0.3411, 0.3434},
         {0.2587, 0.5630, 0.4737, 0.3666, 0.4080, 0.4364, 0.4322, 0.4121},
         {0.4382, 0.5639, 0.6318, 0.4628, 0.5202, 0.6438, 0.5407, 0.3921},
         {0.6248, 0.7637, 0.7166, 0.6804, 0.4853, 0.9119, 0.5899, 0.2844},
         {0.5560, 0.6655, 0.5234, 0.7042, 0.5139, 0.5786, 0.6414, 0.2940},
         {0.4102, 0.5202, 0.3395, 0.5256, 0.6485, 0.3121, 0.6952, 0.4995},
         {0.5434, 0.6769, 0.3828, 0.4559, 0.5879, 0.5411, 0.7154, 0.4636},
         {0.4283, 0.5815, 0.3405, 0.4683, 0.3830, 0.4529, 0.5829, 0.3183}}};

TEST(ADAPTIVEPOOL,forward1)
{
    easynn::Mat input(12,12,3);
    easynn::Mat out(1,1,3);
    input.fillFromArray(ada_input_data);
    out.fillFromArray(ada_out_data1);

    easynn::AdaptivePool adapool;
    adapool.output_size.push_back(1);
    adapool.output_size.push_back(1);
    easynn::Optional option;

    easynn::Mat m1;
    adapool.forward(input,m1,option);
    EXPECT_EQ(compareMat(out,m1),0);
}

TEST(ADAPTIVEPOOL,forward2)
{
    easynn::Mat input(12,12,3);
    easynn::Mat out(2,2,3);
    input.fillFromArray(ada_input_data);
    out.fillFromArray(ada_out_data2);

    easynn::AdaptivePool adapool;
    adapool.output_size.push_back(2);
    adapool.output_size.push_back(2);
    easynn::Optional option;

    easynn::Mat m1;
    adapool.forward(input,m1,option);
    EXPECT_EQ(compareMat(out,m1),0);

}

TEST(ADAPTIVEPOOL,forward3)
{
    easynn::Mat input(12,12,3);
    easynn::Mat out(2,3,3);
    input.fillFromArray(ada_input_data);
    out.fillFromArray(ada_out_data3);

    easynn::AdaptivePool adapool;
    adapool.output_size.push_back(3);
    adapool.output_size.push_back(2);
    easynn::Optional option;

    easynn::Mat m1;
    adapool.forward(input,m1,option);
    EXPECT_EQ(compareMat(out,m1),0);

}



TEST(ADAPTIVEPOOL,forward4)
{
    easynn::Mat input(12,12,3);
    easynn::Mat out(8,11,3);
    input.fillFromArray(ada_input_data);
    out.fillFromArray(ada_out_data4);

    easynn::AdaptivePool adapool;
    adapool.output_size.push_back(11);
    adapool.output_size.push_back(8);
    easynn::Optional option;

    easynn::Mat m1;
    adapool.forward(input,m1,option);
    EXPECT_EQ(compareMat(out,m1),0);

}

TEST(ADAPTIVEPOOL,forward5)
{

    easynn::Mat m1(3,5);
    std::vector<std::vector<float>> x1={{-1,2,-2},{-2,-98,100},{3,-4,5},{-31,4,52},{3,-43,59}};
    m1.fillFromArray(x1);

    easynn::Mat out(1,1);
    std::vector<std::vector<float>> x2={{3.1333}};
    out.fillFromArray(x2);

    easynn::AdaptivePool adapool;
    adapool.output_size.push_back(1);
    adapool.output_size.push_back(1);
    easynn::Optional option;

    easynn::Mat m2;
    adapool.forward(m1,m2,option);
    EXPECT_EQ(compareMat(m2,out),0);

}