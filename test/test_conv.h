#include"test_ulti.h"
#include"test_fun.h"
#include"ir.h"
#include"net.h"
#include"mat.h"

std::vector<std::vector<std::vector<std::vector<float>>>> input_data=\
        {{{{0.2295, 0.2599, 0.6441, 0.0168, 0.7927, 0.3168, 0.9468, 0.2946,
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
           0.1323, 0.3780, 0.7844, 0.0424}}}};

std::vector<std::vector<std::vector<std::vector<float>>>> out_data = \
        {{{{-0.0754, -0.2237, -0.0262,  0.0480, -0.0727, -0.1734,  0.0382,
           -0.2142, -0.2282, -0.1670},
          {-0.2160, -0.3609, -0.2223, -0.0724, -0.1837, -0.1930,  0.1487,
           -0.0935, -0.1778, -0.4095},
          {-0.2878,  0.0353, -0.0875, -0.2558, -0.1071, -0.4258, -0.1939,
            0.0205, -0.1449, -0.2091},
          {-0.0939, -0.3457, -0.0836,  0.1239, -0.0420, -0.0594,  0.0891,
           -0.1130, -0.1706, -0.5745},
          {-0.1072,  0.1951, -0.1803, -0.3547, -0.1862, -0.3887, -0.1535,
            0.2309, -0.1685, -0.2494},
          {-0.0689, -0.3844, -0.2404,  0.0143, -0.1366,  0.2770, -0.0979,
            0.0831, -0.1735,  0.1783},
          { 0.0193,  0.0491, -0.2119,  0.2580, -0.1239, -0.3541, -0.2682,
           -0.1974,  0.1195, -0.4582},
          {-0.1336, -0.0391, -0.0976, -0.2880, -0.1475, -0.3847,  0.0582,
            0.2520, -0.1219,  0.0190},
          { 0.3006,  0.0264, -0.2257, -0.2263, -0.0080, -0.1292, -0.1002,
           -0.2176,  0.0579,  0.0354},
          {-0.1361, -0.3120, -0.4098, -0.2033, -0.2227,  0.0516, -0.2396,
           -0.0877, -0.2232, -0.3101}},

         {{-0.1346, -0.0437,  0.3849,  0.0735,  0.6789,  0.0677,  0.5025,
            0.2014,  0.5297,  0.0100},
          { 0.2256,  0.4630,  0.4059, -0.0493,  0.3361,  0.1011,  0.3677,
            0.5620,  0.4297,  0.3978},
          { 0.3495,  0.1810,  0.4347,  0.4148,  0.4078,  0.3537,  0.3435,
            0.2580,  0.2200,  0.2378},
          { 0.3520,  0.2566,  0.4001,  0.3477,  0.3476,  0.3712,  0.3044,
            0.4033,  0.1817,  0.3777},
          { 0.1504,  0.2988,  0.2552,  0.4125,  0.4302,  0.3593, -0.0856,
            0.3499,  0.3533,  0.5450},
          { 0.1905,  0.2547,  0.3744,  0.5179,  0.1430,  0.1794,  0.1764,
            0.3613,  0.5693,  0.3688},
          { 0.4976,  0.3808,  0.3558,  0.1958,  0.3729,  0.2781,  0.4666,
            0.3366,  0.2823,  0.4841},
          { 0.0132,  0.6507,  0.3071,  0.5733,  0.5467,  0.3240,  0.2739,
            0.3595,  0.2753,  0.5820},
          { 0.3531,  0.4957,  0.3602,  0.1418,  0.2319,  0.2237,  0.3659,
           -0.1160,  0.5537,  0.2505},
          { 0.1175,  0.1157,  0.3067,  0.0553,  0.3657,  0.0607,  0.2903,
            0.2158,  0.3469,  0.2586}},

         {{-0.3459, -0.3202, -0.1616, -0.1540, -0.0093, -0.0420, -0.2537,
           -0.0279, -0.0308, -0.1473},
          {-0.2104, -0.1863,  0.1329, -0.3662, -0.1729, -0.2341, -0.2179,
            0.0577, -0.1930, -0.1813},
          { 0.0653, -0.0658,  0.0078, -0.2219,  0.1197, -0.5254, -0.0450,
           -0.1164,  0.0232, -0.1911},
          {-0.0936, -0.1640,  0.0067,  0.1356, -0.0108, -0.1386,  0.0651,
           -0.2826, -0.2384, -0.2962},
          { 0.0866, -0.0697, -0.2974, -0.1706,  0.0312, -0.0995, -0.1990,
           -0.1156,  0.0068, -0.0418},
          {-0.4030, -0.3581,  0.0714, -0.0955, -0.0626,  0.3599, -0.3416,
           -0.1331, -0.0140,  0.0286},
          { 0.0370, -0.0955, -0.1521,  0.0264, -0.4718, -0.0750, -0.2645,
            0.0503, -0.3442, -0.0781},
          {-0.2190,  0.1145, -0.2420, -0.0637,  0.1724, -0.4259,  0.1265,
           -0.0685, -0.0903,  0.1716},
          { 0.0199, -0.0608,  0.1309, -0.2531,  0.0080, -0.1949, -0.0918,
           -0.3634,  0.2270, -0.1858},
          {-0.4058, -0.4360, -0.3671, -0.1427, -0.4255, -0.0408, -0.3916,
           -0.2498, -0.1944, -0.3262}}}};
        
TEST(layer,conv_loadParam)
{
    easynn::Net net;
    EXPECT_EQ(net.loadModel("/home/hp/code/github/EasyNN/example/conv.pnnx.param",\
    "/home/hp/code/github/EasyNN/example/conv.pnnx.bin"),0);
    easynn::Mat input(12,12,3,1);
    easynn::Mat output(10,10,3,1);
    input.fillFromArray(input_data);
    output.fillFromArray(out_data);

    net.input(0,input);
    easynn::Mat m;
    net.extractBlob(1,m);
    printMat(m); 
}

