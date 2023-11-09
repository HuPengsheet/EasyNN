#include "blob.h"

namespace easynn {

Blob::Blob()
{
    producer = -1;
    consumer = -1;
    shape = Mat();
}

} // namespace easynn
