#include <stdio.h>

int square(int i) {
	return i * i;
}

int listfn(float array[]) {
    for (int i = 0; i < array.length; i++) {
        printf(array[i]);
    }

}

// for f in overlap:
        // print( f[-1] )
    // try:
        // n = float( f[-1] )
        // if n <= 1 and n >= 0:
            // coverage_vec.append(n)
    //  except:
        // n = 0
        // coverage_vec.append(n)
