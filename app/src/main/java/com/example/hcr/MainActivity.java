package com.example.hcr;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.hcr.ml.Model;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

public class MainActivity extends AppCompatActivity {

    private ImageView imgView;
    private Button select, predict;
    private TextView tv;
    private Bitmap img;

    private static final int BATCH_SIZE = 1;
    public static final int IMG_HEIGHT = 28;
    public static final int IMG_WIDTH = 28;
    private static final int NUM_CHANNEL = 1;
    private  ByteBuffer mImageData;
    public static float total =0;
    private  int[] mImagePixels;
    Mat imageMat;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        OpenCVLoader.initDebug();
        imgView = (ImageView) findViewById(R.id.imageView);
        tv = (TextView) findViewById(R.id.textView);
        select = (Button) findViewById(R.id.button);
        predict = (Button) findViewById(R.id.button2);



        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 100);

            }
        });

        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {

                    img = Bitmap.createScaledBitmap(img, 28, 28, true);


                    imageMat = new Mat();
                    Utils.bitmapToMat(img, imageMat);
                    //Imgproc.resize(imageMat, imageMat, new Size(28,28 ));

                    //Imgproc.threshold(imageMat, imageMat, 120, 255,Imgproc.THRESH_BINARY);
                    //Bitmap bmp = Bitmap.createBitmap(imageMat.cols(), imageMat.rows(), Bitmap.Config.ARGB_8888);
                    Imgproc.cvtColor(imageMat, imageMat, Imgproc.COLOR_BGR2GRAY);
                    Utils.matToBitmap(imageMat, img);
                    //img = bmp;
                    imgView.setImageBitmap(img);
                    Log.d("leng" , String.valueOf(img.getHeight()));

                }catch (Exception e){}


                try {
                    Model model = Model.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 28, 28, 1}, DataType.FLOAT32);

                    mImagePixels = new int[IMG_HEIGHT * IMG_WIDTH];
                    mImageData = ByteBuffer.allocateDirect(
                            4 * BATCH_SIZE * IMG_HEIGHT * IMG_WIDTH * NUM_CHANNEL);
                    convertBitmapToByteBuffer1(img);

                   /* TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                    tensorImage.load(img);
                    ByteBuffer byteBuffer = tensorImage.getBuffer();*/

                    inputFeature0.loadBuffer(mImageData);

                    // Runs model inference and gets result.
                    Model.Outputs outputs = model.process(inputFeature0);

                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    // Releases model resources if no longer used.
                    model.close();


                    float f[] = outputFeature0.getFloatArray();
                    int label = 0;
                    float maxVal = (float) 0.0;
                    Log.d("test1", String.valueOf(f.length));
                    for(int i =0 ; i < f.length ; i++){
                        if( f[i] > maxVal){
                            maxVal = f[i];
                            label = i;
                        }
                    }
                    Log.d("test", String.valueOf(img.getByteCount()));
                    Log.d("test1", String.valueOf(maxVal));

                    tv.setText("Label :" + label);


                } catch (IOException e) {
                    // TODO Handle the exception
                }

            }
        });

    }

    private void convertBitmapToByteBuffer1(Bitmap bitmap) {
        if (mImageData == null) {
            return;
        }
        mImageData.rewind();


        bitmap.getPixels(mImagePixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        // Convert the image to floating point.
        int pixel = 0;

        for (int i = 0; i < IMG_WIDTH; ++i) {
            for (int j = 0; j < IMG_HEIGHT; ++j) {
                final int val = mImagePixels[pixel++];

                float temp = ((((val >> 16) & 0xFF)
                        + ((val >> 8) & 0xFF)
                        + (val & 0xFF) )) / 255.0f;

                Log.d("value", String.valueOf(imageMat.get(i,j)[0]));

                mImageData.putFloat(temp);


                //mImageData.putFloat((temp-59.93674599535272f)/94.49973174906522f);
                //mImageData.putFloat(((val>> 16) & 0xFF) / 255.f);
                //mImageData.putFloat(((val>> 8) & 0xFF) / 255.f);
                //mImageData.putFloat((val & 0xFF) / 255.f);
            }
        }

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == 100)
        {
            imgView.setImageURI(data.getData());

            Uri uri = data.getData();
            try {
                img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }





    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (mImageData == null) {
            Log.d("test","found");
            return;
        }
        mImageData.rewind();
        total=0;
        bitmap.getPixels(mImagePixels, 0, bitmap.getWidth(), 0, 0,
                bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < IMG_WIDTH; ++i) {
            for (int j = 0; j < IMG_HEIGHT; ++j) {
                int value = mImagePixels[pixel++];
                //mImageData.putFloat(convertPixelWhite(value));
                total = total + convertPixelWhite(value);
            }
        }
        bitmap.getPixels(mImagePixels, 0, bitmap.getWidth(), 0, 0,
                bitmap.getWidth(), bitmap.getHeight());
        pixel=0;
        if(total>800) {
            for (int i = 0; i < IMG_WIDTH; ++i) {
                for (int j = 0; j < IMG_HEIGHT; ++j) {
                    int value = mImagePixels[pixel++];
                    mImageData.putFloat(convertPixelWhite(value));

                }
            }
        }
        else{
            for (int i = 0; i < IMG_WIDTH; ++i) {
                for (int j = 0; j < IMG_HEIGHT; ++j) {
                    int value = mImagePixels[pixel++];
                    mImageData.putFloat(convertPixelBlack(value));

                }
            }
        }
    }

    private static float convertPixelWhite(int color) {
        float color2 = ((((color >> 16) & 0xFF) * 0.299f
                + ((color >> 8) & 0xFF) * 0.587f
                + (color & 0xFF) * 0.114f)) / 255.0f;
        if(color2 < 0.6f)
            return 0.0f;
        else
            return 1.0f;
    }
    private static float convertPixelBlack(int color) {
        float color2 = (255.0f - (((color >> 16) & 0xFF) * 0.299f
                + ((color >> 8) & 0xFF) * 0.587f
                + (color & 0xFF) * 0.114f)) / 255.0f;
        if(color2 < 0.6f)
            return 0.0f;
        else
            return 1.0f;
    }
}