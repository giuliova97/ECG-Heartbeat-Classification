import sys
import FStart
import VideoClassStart
import VMakerStart


if sys.argv[1] == '0':
    FStart.start(sys.argv[2])
elif sys.argv[1] == '1':
    VideoClassStart.start()
elif sys.argv[1] == '2':
    VMakerStart.start(sys.argv[2])