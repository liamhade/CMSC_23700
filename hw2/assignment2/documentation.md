Please do not grade this submission yet, I am using my late bank.

1.

(a)
For the orthographic project, n > f in general because we have the w axis pointing away from the object.

(b)
Our x-values lies between our r and l values. Our y-values between our t and b values. And our z-values between our n and f values. For each of our x, y, and z values, we want to scale them to be between [-1, 1]. Thus, we want to apply a linear transfom to move whatever values our coordinates are in, into the length-2 range between [-1, 1]. Moreover, we want to make sure that our value for each coordinate is centered at 0, so we want to shifted everyting over by half the length of the previous range (e.g. (r - l)/2 for x). Thus, our canonical coordinates for x, y, and z are:


x_{canonical} = 2/(r - l) * (x_{ortho} - (r + l)/2)
y_{canonical} = 2/(t - b) * (y_{ortho} - (t + b)/2)
z_{canonical} = 2/(n - f) * (z_{ortho} - (n + f)/2)

Thus, our orthographic matrix is:

M_{ortho} = [
	[2/(r - l), 0, 0, -(r + l)/(r - l)],
	[0, 2/(t - b), 0, -(t + b)/(t - b)],
	[0, 0, 2/(n - f), -(n + f)/(n - f)],
	[0, 0, 0, 1]
]

2.

P = [
	[n, 0, 0, 0],
	[0, n, 0, 0],
	[0, 0, n+f, -fn],
	[0, 0, 1, 0]
]

v1 = [x1 y1 z1 1]
v2 = [x2 y2 z2 1]

Pv1 =  [nx1 ny1 z1(n+f)-fn z1]
Pv2 =  [nx2 ny2 z2(n+f)-fn z2]

After applying our perspective divide and isolating the z-value, we get:

z1' = n+f - fn/z1
z2' = n+f - fn/z2

If we assume that z1 < z2, then -fn/z1 will be more negative than - fn/z2,
and thus z1' < z2.

---------------------------------

My initial approach was to write down all the equation from the slides and the textbook and keep it in a seperate Google Doc for easy viewing. 

During implementation, I had some trouble figuring out hhow to seperate the functions for barycentric coordinates into a sensible way. Moreover, I realized halfway trough that I didn't understand what the values for alpha, beta, and gamma were. I eventually realized that they are not general values for a triangle, but for a triangle AND a point.

Overall, the largest help for me was going to office hours (thank you Ron Kiehn!) and sticking to the textbook.

Used this source to find a matrix way of finding if a point is inside a triangle: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates.html