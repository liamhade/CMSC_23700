Realized that I wasn't calculating the transformation between viewbox and image coordinates correctly.

Realized that I wasn't indexing into my image properly (x and y coordinates were flipped).

Realized that I need to convert from viewbox to image coordinates first before doing supersampling.

Order is very important.