import argparse
import cv2
from inference import Network
import send_mail

count = 100
INPUT_STREAM = "trespasser.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():

    parser = argparse.ArgumentParser("Run inference on an input video")

    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    ct_desc = "The confidence threshold to use with the bounding boxes"

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')


    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default='BLUE')
    optional.add_argument("-ct", help=ct_desc, default=0.5)
    args = parser.parse_args()

    return args

def convert_color(color_string):

    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']


def draw_boxes(frame, result, args, width, height):

    global count
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.ct:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, 3)
            cv2.imwrite("trespassers.png",frame)
            count += 1
        if count > 100:
            send_mail.main()
            count = 0
        
    return frame

def infer_on_video(args):
    args.c = convert_color(args.c)
    args.ct = float(args.ct)
    plugin = Network()

    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)
    
    width = int(cap.get(3))
    height = int(cap.get(4))

    out = cv2.VideoWriter('trespasser_detected.mp4', 0x00000021, 30, (width,height))
    

    while cap.isOpened():

        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        plugin.async_inference(p_frame)

        if plugin.wait() == 0:
            result = plugin.extract_output()
            frame = draw_boxes(frame, result, args, width, height)
            out.write(frame)

        if key_pressed == 27:
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
