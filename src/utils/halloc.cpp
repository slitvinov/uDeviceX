#include <stdlib.h>

#include "halloc.h"

int emalloc(size_t size, /**/ void **data) {
    *data = malloc(size);

    if (NULL == *data) {
        // TODO: use ERROR 
        printf("Could not allocate array of size %ld\n", size);
        return 1;
    }
    return 0;
}
