## Initial Approach

My initial approach on this project was to write everything -- math, function signatures, and conceptual
explanations -- down on paper. The first function is constructed was to shift between the viewbox coordinates
and the image coordinates. Next, I had to figure out how I would test whether a point was inside or outside a shape.

This part of the initial process was the trickiest and most time consuming. Even though during this part of the design process I was focusing mainly on how to check if a point lied within a triangle, I needed the method to be generalizable so that it could work for as many shapes as possible. Morever, I knew that once I settled on a method, it would have many downstream side-effects and implications for how the rest of the code was written. My final method for figuring out if a point lied within a shape (i.e. triangle or line) was to create a series of lines between each of the defining points of the shape (or, in the case of the line, I had to first figure out where each corner of the line was by using it's edge coordinates and width). Once I had these boundary lines, I used the points of the shape to determine which side of the boundary line would count as "inside".

Each boundary line then had an inside and outside, and determining whether a point lied inside the shape simply meant checking that it was contained within all of the boundary lines. In my code, the functions for checking if a point lies within a region are called "region_testers". This process was only neccessary for triangles and lines, since the equation for a circle made it much simpler to check if a point was contained within it.

Once I had my function for checking if a point lied within a shape, calculating the coverage for a pixel was the next logical move (altough in practice, I actually started with a skeleton for the coverage function and worked backwards, then came back to it later). Determining the coverage was rather straightforward, since I merely had to use the point_in_shape() functions that I already had to see what fraction of the 9 sample points of the pixel were contained in the shape.

I should also note that I started off the coding / thought process assuming that anti-alias would be turned on, since I figured it'd be easier to turn it off once I had the more complex anti-aliasing, as opposed to make adding on to a simpler non-anti-aliasing model later. When anti-aliasing is turned off, my rudimentary approach for checking if a pixel is in/out of a shape is simply to check the upper-left corner of the pixel.

## Challenges

One of the challenges I came across was how to handle the viewbox to image coordinate translation. I settled on an approachh that uses a function that finds the aspect ratio between the viewbox and image width, and the viewbox and image height, and uses these values to scale the x-coordinates and y-coordinates. Using this scaling factor was easy to implement for the triangles, since I just had to apply the scaling to each of the corners. 

Realized that I wasn't indexing into my image properly (x and y coordinates were flipped).

Realized that I need to convert from viewbox to image coordinates first before doing supersampling.

Order is very important.

## Takeaways

## Sources

ChatGPT was used as an assistant for this project to explain conceptual questions,
as well as to aid at times in the debugging process. A link to the chat can be found below:
https://chatgpt.com/share/67f1607a-a69c-8005-8570-3edf470249c3