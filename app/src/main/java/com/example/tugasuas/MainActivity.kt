package com.example.tugasuas

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.tensorflow.lite.Interpreter

class MainActivity : AppCompatActivity() {

    private val TAG = "ASLProject"
    private val MODEL_PATH = "asl_newmodel.tflite"
    private val LABEL_PATH  = "labels.txt"

    private var tflite: Interpreter? = null
    private lateinit var labels: Array<String>
    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView
    private lateinit var confidenceTextView: TextView
    private var imageCapture: ImageCapture? = null
    private lateinit var previewView: PreviewView
    private lateinit var cameraExecutor: ExecutorService

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        resultTextView = findViewById(R.id.resultTextView)
        confidenceTextView = findViewById(R.id.confidenceTextView)
        val captureButton: Button = findViewById(R.id.captureButton)
        previewView = findViewById(R.id.previewView)

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Check for camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            val requestPermissionLauncher =
                registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
                    if (isGranted) {
                        startCamera()
                    } else {
                        Log.e(TAG, "Camera permission denied.")
                    }
                }
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        } else {
            startCamera()
        }

        // Load the TensorFlow Lite model and labels
        try {
            tflite = Interpreter(loadModelFile())
            labels = loadLabels()
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model or labels", e)
            resultTextView.text = "Failed to load model"
            confidenceTextView.text = "Please try again later"
        }

        // Set up capture button
        captureButton.setOnClickListener { takePicture() }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                imageCapture = ImageCapture.Builder()
                    .setTargetRotation(previewView.display.rotation)
                    .build()

                val cameraSelector = CameraSelector.Builder()
                    .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                    .build()

                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)

            } catch (e: Exception) {
                Log.e(TAG, "Error starting camera", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePicture() {
        imageCapture?.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    try {
                        val bitmap = convertImageProxyToBitmap(image)
                        imageView.setImageBitmap(bitmap)

                        // Run inference in background thread
                        val result = runInference(bitmap)
                        resultTextView.text = result[0]
                        confidenceTextView.text = result[1]

                    } catch (e: Exception) {
                        Log.e(TAG, "Error processing image", e)
                    } finally {
                        image.close()
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exception.message}", exception)
                }
            })
    }

    private fun convertImageProxyToBitmap(image: ImageProxy): Bitmap {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        image.close()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun runInference(bitmap: Bitmap): Array<String> {
        val result = arrayOf("Loading...", "0%")
        Thread {
            try {
                val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

                val inputShape = tflite!!.getInputTensor(0).shape()
                val inputDataType = tflite!!.getInputTensor(0).dataType()

                val inputBuffer = TensorBuffer.createFixedSize(inputShape, inputDataType)
                inputBuffer.loadBuffer(convertBitmapToByteBuffer(resizedBitmap))

                val outputShape = tflite!!.getOutputTensor(0).shape()
                val outputDataType = tflite!!.getOutputTensor(0).dataType()

                val outputBuffer = TensorBuffer.createFixedSize(outputShape, outputDataType)

                tflite!!.run(inputBuffer.buffer, outputBuffer.buffer)

                val output = outputBuffer.floatArray

                var maxIndex = 0
                var maxConfidence = 0f
                for (i in output.indices) {
                    if (output[i] > maxConfidence) {
                        maxConfidence = output[i]
                        maxIndex = i
                    }
                }

                // Update UI on the main thread
                runOnUiThread {
                    result[0] = "Label: ${labels[maxIndex]}"
                    result[1] = "Confidence: ${maxConfidence * 100}%"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error running inference", e)
            }
        }.start()
        return result
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = assets.openFd(MODEL_PATH)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabels(): Array<String> {
        return assets.open(LABEL_PATH).bufferedReader().useLines { it.toList() }.toTypedArray()
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val size = 224 * 224 * 3
        val byteBuffer = ByteBuffer.allocateDirect(size * 4)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(224 * 224)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (pixelValue in intValues) {
            byteBuffer.putFloat(((pixelValue shr 16) and 0xFF) / 255.0f)
            byteBuffer.putFloat(((pixelValue shr 8) and 0xFF) / 255.0f)
            byteBuffer.putFloat((pixelValue and 0xFF) / 255.0f)
        }
        return byteBuffer
    }

    override fun onDestroy() {
        super.onDestroy()
        tflite?.close()
        cameraExecutor.shutdown()
    }
}
