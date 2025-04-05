## Initial Approach

My initial approach on this project was to write everything -- math, function signatures, and conceptual
explanations -- down on paper. The first function is constructed was to shift between the viewbox coordinates
and the image coordinates. Next, I had to figure out how I would test whether a point was inside or outside a shape.

This part of the initial process was the trickiest and most time consuming. Even though during this part of the design process I was focusing mainly on how to check if a point lies within a triangle, I needed the method to be generalizable so that it could work for as many shapes as possible. Moreover, I knew that once I settled on a method, it would have many downstream side-effects and implications for how the rest of the code was written. My final method for figuring out if a point lies within a shape (i.e. triangle or line) was to create a series of lines between each of the defining points of the shape (or, in the case of the line, I had to first figure out where each corner of the line was by using it's edge coordinates and width). Once I had these boundary lines, I used the points of the shape to determine which side of the boundary line would count as "inside".

Each boundary line then had an inside and outside, and determining whether a point lies inside the shape simply meant checking that it was contained within all of the boundary lines. In my code, the functions for checking if a point lies within a region are called "region_testers". This process was only necessary for triangles and lines, since the equation for a circle made it much simpler to check if a point was contained within it.

Once I had my function for checking if a point lies within a shape, calculating the coverage for a pixel was the next logical move (although in practice, I actually started with a skeleton for the coverage function and worked backwards, then came back to it later). Determining the coverage was rather straightforward, since I merely had to use the point_in_shape() functions that I already had to see what fraction of the 9 sample points of the pixel were contained in the shape.

I should also note that I started off the coding / thought process assuming that anti-alias would be turned on, since I figured it'd be easier to turn it off once I had the more complex anti-aliasing, as opposed to adding on to a simpler non-anti-aliasing model later. When anti-aliasing is turned off, my rudimentary approach for checking if a pixel is in/out of a shape is simply to check the upper-left corner of the pixel.

## Challenges

One of the challenges I came across was how to handle the viewbox to image coordinate translation. I settled on an approach that uses a function that finds the aspect ratio between the viewbox and image width, and the viewbox and image height, and uses these values to scale the x-coordinates and y-coordinates. Using this scaling factor was easy to implement for the triangles, since I just had to apply the scaling to each of the corners. For the line, it was a bit trickier, since the width couldn't just be multiplied by the scaling factor, since the width can point in both the x and y directions. To get around this issue, I first found the 4 corners of the line in the viewbox coordinates, then translated this to the image coordinates. For translating viewbox to image coordinates for the circle, I had to add two extra parameters so I could squish the circle if the aspect ratio needed to be changed. This essentially converted the circle into an ellipse.

Another issue that cropped up, but which was easily amended, was that I was incorrectly indexing into the image using img[x,y], instead of img[y,x], since I forgot that the columns are indexed first, then the rows.

Overall, one of the biggest challenges was making sure that the conversion from viewbox to image translation made sense and was used consistently in the program. In earlier iterations of my code, I often accidently plotted the point in the correct image coordinate, but then by supersampling wasn't scaled, so the blurring was off. Little translation issues like that cropped up time and time and again.

## Takeaways

For a big project like this that uses a lot of math, I found it a necessity to put all my thoughts on paper first before I moved to the computer. Once I felt confident that my math and architecture would sensibly translate to code, only then did my fingers touch the keyboard. Moreover, having good documentation for each function was very helpful for the debugging process, since it kept a log of the types of variables I needed to pass in. And lastly, I had a dedicated testing.py function that I used to import all the functions from raster.py to test errors where I suspected there to be a bug.

For anyone embarking on coding a rasterizer in the future, I would first suggest that they hold back from coding for even longer than I did -- make sure that all the math checks out on paper first (knowing when to translate between coordinate systems, or how you'll check if a point is in an arbitrary shape, etc...). For bigger projects, the code often has a tendency to grow in unforeseen ways, which only makes the debugging process harder. Starting on paper and pencil gives you the chance to think more abstractly and freely, in the hopes of coming up with a cleaner solution than you otherwise would have.

And lastly, remember to make your code open to changes. I started my program with anti-aliasing already built in, but if you started without it, make sure that the architecture of the code would be modular enough for anti-aliasing (or some other similar function) to be easily implemented.

## Sources

ChatGPT was used as an assistant for this project to explain conceptual questions,
as well as to aid at times in the debugging process. A link to the chat can be found below:
https://chatgpt.com/share/67f1607a-a69c-8005-8570-3edf470249c3

