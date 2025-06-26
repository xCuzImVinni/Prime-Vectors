# Prime-Vectors
I’m building a high-performance arithmetic system in Rust that rethinks number representation, by encoding integers as prime exponent vectors.
I optimized the standard binary GCD/LCM algorithm with a combination of my method and the usual method, it's slightly faster, but sadly I can't solely use my Prime Factor System (PFS) as the integer factorization algorithms are way too slow and often lays in NP - with quantum computers Shor's algorithm could become viable though combined with my method, especially for really large numbers!
If I didn't have to convert and reconvert the numbers to PFS and back to decimal/binary it would be even faster for many more operations, such as basic multiplication and division of whole numbers that divide each other.
Also added a GUI with some data to compare the algorithms on a range of 4-32 bit for 1000 iterations.
