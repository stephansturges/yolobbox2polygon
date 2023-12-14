Convert YOLO bboxes to polygons automatically using SAM!

You will need the SAM-HQ-vit-h weights from here https://huggingface.co/lkeab/hq-sam/tree/main, put the file in the root here.

Install dependencies.

Put your YOLO bbox-format dataset in ./yolo_dataset/images/ and ./yolo_dataset/labels/

Run the python script. 

And you're done! Nice clean bounding polygons from your bounding boxes.

Merry Christmas!



Copyright 2023 Stephan Sturges

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
I PROBABLY LIFTED HALF THIS CODE FROM SOMEWHERE ELSE, SO THANK YOU.
