#ifndef EASYNN_BLOB_H
#define EASYNN_BLOB_H

#include "mat.h"


namespace easynn {

class  Blob
{
public:
    // empty
    Blob();

public:

    int producer;
    int consumer;
    Mat shape;
};

} // namespace

#endif 