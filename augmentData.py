
############################ LIBRARIES ############################

import tensorflow as tf
import numpy as np
from random import randint, random
import cv2


############################ PARAMETERS ############################


image_width = 150
image_height = 110
SAMPLE_RATE = 16000
NUM_MEL_BINS = 64
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01 
# Mel spectrum constants and functions.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


############################# FUNCTIONS ############################


def central_scale_images(image, scale):

    x1 = y1 = 0,5 - 0,5*scale
    x2 = y2 = 0,5 + 0,5*scale
    box = np.array([y1, x1, y2, x2])

    crop_size = np.array([image_height, image_width], dtype=np.float32)

    scaled_img = tf.image.crop_and_resize(image, box, 1, crop_size)

    return scaled_img


def rotate_images(image, angle):

    radian = angle * np.pi / 180
    rotated_img = tf.contrib.image.rotate(image, radian)

    return rotated_img


def flip_images(image):

    flipped_img = tf.image.flip_left_right(image)

    return flipped_img


def translate_images(image, offset):

    image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.reshape(image,[1,tf.shape(image)[0],tf.shape(image)[1],3])
    offset_2 = np.array([offset], dtype=np.float32)
    size = np.array([int(np.ceil((1-abs(offset[1]))*224)), int(np.ceil((1-abs(offset[0]))*224))], dtype=np.int32)
    '''
    if offset[0] == 0:

        if offset[1] > 0:

            w_start = h_start = 0
            h_end = IMAGE_SIZE
            w_end = int(np.ceil((1-offset[1])*IMAGE_SIZE))

        elif offset[1] < 0:

            w_end = h_end = IMAGE_SIZE
            h_start = 0
            w_start = int(np.floor(abs(offset[1])*IMAGE_SIZE))

    elif offset[1] == 0:

        if offset[0] > 0:

            w_start = h_start = 0
            w_end = IMAGE_SIZE
            h_end = int(np.ceil((1-offset[0])*IMAGE_SIZE))

        elif offset[0] < 0:

            w_end = h_end = IMAGE_SIZE
            w_start = 0
            h_start = int(np.floor(abs(offset[0])*IMAGE_SIZE))
    '''
    glimps = tf.image.extract_glimpse(image, size, offset_2)
    
    translated_img = tf.image.resize_images(glimps,[224,224],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    translated_img = tf.reshape(translated_img, [224,224,3])
    translated_img = tf.image.convert_image_dtype(translated_img,tf.uint8)

    return translated_img


def augment_face1(image, face_case, seed):
    '''
    if seed >= 0.5:
        offset = 0.1
    else:
        offset = 0.2
    '''
    if face_case == 1:

        angle = int(round(seed*30))
        image = rotate_images(image, angle)

    elif face_case == 2:

        angle = int(round(seed*30))
        image = rotate_images(image, -angle)

    elif face_case == 3:

        image = flip_images(image)
    
    elif face_case == 4:

        offset = seed*0.2
        image = translate_images(image, np.array([-offset, offset]))

    elif face_case == 5:

        offset = seed*0.2
        image = translate_images(image, np.array([-offset, -offset]))

    elif face_case == 6:

        offset = seed*0.2
        image = translate_images(image, np.array([offset, offset]))

    elif face_case == 7:

        offset = seed*0.2
        image = translate_images(image, np.array([offset, -offset]))
    
    elif face_case == 8:

        image = tf.image.flip_up_down(image)

    elif face_case == 9:

        image = tf.image.random_brightness(image, max_delta=0.5)

    return image

def augment_face(image, angle):

    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=25/255)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.contrib.image.rotate(image, angle * np.pi / 180, interpolation='BILINEAR')

    image = tf.random_crop(image, [200,200,3])
    image = tf.image.resize_images(image,[224,224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        

    return image

'''
def false_fn(warper, step):

    warper = float(1)

    return warper


def true_fn(warper, step):

    i = tf.constant(1, dtype=tf.int64)

    c = lambda r: tf.less(step, 10000*i)
    b = lambda i: tf.add(i,1)
    tf.while_loop(c, b, [i])
    
    warper = warper[1]

    return warper
'''

def find_warper(warper, step):

    step = tf.cast(step, tf.int64)
    t = 1
    i = tf.constant(1, dtype=tf.int64)
    c = lambda r: tf.less(step, 10000*i)
    b = lambda i: tf.add(i,1)
    tf.while_loop(c, b, [i])
    with tf.Session() as sess:
        
        i = i.eval()
    
    warper = warper[i-1]

    return warper


def hertz_to_mel(frequencies_hertz):
    """
    Convert frequencies to mel scale using HTK formula.
    Args:
        frequencies_hertz: Scalar or np.array of frequencies in hertz.
    Returns:
        Object of same size as frequencies_hertz containing corresponding values
        on the mel scale.
    """
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def spectrogram_to_mel_matrix(warper=1, num_mel_bins=20,
                              num_spectrogram_bins=129,
                              audio_sample_rate=8000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=3800.0):
    """
    Return a matrix that can post-multiply spectrogram rows to make mel.
    Returns a np.array matrix A that can be used to post-multiply a matrix S of
    spectrogram values (STFT magnitudes) arranged as frames x bins to generate a
    "mel spectrogram" M of frames x num_mel_bins.  M = S A.
    The classic HTK algorithm exploits the complementarity of adjacent mel bands
    to multiply each FFT bin by only one mel weight, then add it, with positive
    and negative signs, to the two adjacent mel bands to which that bin
    contributes.  Here, by expressing this operation as a matrix multiply, we go
    from num_fft multiplies per frame (plus around 2*num_fft adds) to around
    num_fft^2 multiplies and adds.  However, because these are all presumably
    accomplished in a single call to np.dot(), it's not clear which approach is
    faster in Python.  The matrix multiplication has the attraction of being more
    general and flexible, and much easier to read.
    Args:
        num_mel_bins: How many bands in the resulting mel spectrum.  This is
            the number of columns in the output matrix.
        num_spectrogram_bins: How many bins there are in the source spectrogram
            data, which is understood to be fft_size/2 + 1, i.e. the spectrogram
            only contains the nonredundant FFT bins.
        audio_sample_rate: Samples per second of the audio at the input to the
            spectrogram. We need this to figure out the actual frequencies for
            each spectrogram bin, which dictates how they are mapped into mel.
        lower_edge_hertz: Lower bound on the frequencies to be included in the mel
            spectrum.  This corresponds to the lower edge of the lowest triangular
            band.
        upper_edge_hertz: The desired top edge of the highest frequency band.
    Returns:
        An np.array with shape (num_spectrogram_bins, num_mel_bins).
      Raises:
        ValueError: if frequency edges are incorrectly ordered or out of range.
    """
    nyquist_hertz = audio_sample_rate / 2.
    if lower_edge_hertz < 0.0:
        raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                     (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > nyquist_hertz:
        raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" %
                     (upper_edge_hertz, nyquist_hertz))

    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    
    for spec_bin in spectrogram_bins_hertz:

        if spec_bin <= upper_edge_hertz:

            spec_bin *= warper
        
        else:

            spec_bin = nyquist_hertz - (nyquist_hertz-upper_edge_hertz*min(warper,1))*(nyquist_hertz-spec_bin)/(nyquist_hertz-upper_edge_hertz*min(warper,1)/warper)

    
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
    # The i'th mel band (starting from i=1) has center frequency
    # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
    # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
    # the band_edges_mel arrays.
    band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                               hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)

    # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
    # of spectrogram values.
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))

    for i in range(num_mel_bins):

        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
        # Calculate lower and upper slopes for every spectrogram bin.
        # Line segments are linear in the *mel* domain, not hertz.
        lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                   (center_mel - lower_edge_mel))
        upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                   (upper_edge_mel - center_mel))
        # .. then intersect them with each other and zero.
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                          upper_slope))
    # HTK excludes the spectrogram DC bin; make sure it always gets a zero
    # coefficient.
    mel_weights_matrix[0, :] = 0.0

    return mel_weights_matrix.astype(np.float32)


def augment_spectrogram(spectrogram, warper):

    mel_spectrogram = tf.tensordot(spectrogram, tf.convert_to_tensor(spectrogram_to_mel_matrix(
        warper=warper,
        num_spectrogram_bins=np.shape(spectrogram)[1],
        audio_sample_rate=SAMPLE_RATE,
        num_mel_bins=NUM_MEL_BINS,
        lower_edge_hertz=MEL_MIN_HZ,
        upper_edge_hertz=MEL_MAX_HZ)), axes=1 )

    return tf.log(mel_spectrogram + LOG_OFFSET)


def main(unused_argv):

    '''
    imagePATH='/home/gryphonlab/Ioannis/Works/BAUM-1/InputFaces/BAUM1s_MP4_all/S021_077/frame42.jpg'
    image_string = tf.read_file(imagePATH)
    original = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(original,tf.float32)
    #image = tf.reshape(original,[image_width,image_height,3])
    
    image = tf.image.resize_images(image,[224,224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = augment_face(image, 6, 1)
    image = tf.image.convert_image_dtype(image,tf.uint8)
    image = tf.image.encode_jpeg(image,format='rgb')

    fwrite = tf.write_file('/home/gryphonlab/Ioannis/Works/CODES/RML/99.jpeg', image)
    sess = tf.Session().run(fwrite)
    '''
    a=1
    spectrogramPATH = '/home/gryphonlab/Ioannis/Works/RML/InputSpectrograms/s1/an1/t_1.9500625.npy'
    spectrogram_string = tf.read_file(spectrogramPATH)
    spectrogram_decoded = tf.decode_raw(spectrogram_string,tf.float32, little_endian=True)
    spectrogram_decoded = spectrogram_decoded[32:]
    spectrogram_decoded = tf.reshape(spectrogram_decoded,[96,257])
    spectrogram_decoded = augment_spectrogram(spectrogram_decoded, a)
    #spectrogram_decoded = tf.image.convert_image_dtype(spectrogram_decoded,tf.uint8)
    #spectrogram_decoded = tf.encode_raw(spectrogram_decoded)
    

    #with tf.Session() as sess:

    #    saver = tf.train.Saver([spectrogram_decoded])

    #    saver.save(sess,'/home/gryphonlab/Ioannis/Works/CODES/RML/my-checkpoint')
    np.save('/home/gryphonlab/Ioannis/Works/CODES/RML/98.npy',tf.Session().run(spectrogram_decoded),allow_pickle=False)




if __name__ == '__main__':
    tf.app.run()

