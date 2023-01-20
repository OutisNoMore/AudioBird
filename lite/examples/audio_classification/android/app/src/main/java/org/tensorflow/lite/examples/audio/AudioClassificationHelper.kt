/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.audio

import android.content.Context
import android.media.AudioRecord
import android.os.SystemClock
import android.util.Log
import java.util.concurrent.ScheduledThreadPoolExecutor
import java.util.concurrent.TimeUnit
import org.tensorflow.lite.examples.audio.fragments.AudioClassificationListener // in AudioFragment
import org.tensorflow.lite.support.audio.TensorAudio
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import org.tensorflow.lite.task.core.BaseOptions

class AudioClassificationHelper(
  // member variables - initialized at bottom
  val context: Context,                                   // context(phone) to run model in
  val listener: AudioClassificationListener,              // listen for results from tflite model
  var currentModel: String = YAMNET_MODEL,                // path to tflite model to run - yamnet or speech
  var classificationThreshold: Float = DISPLAY_THRESHOLD, // acceptable error for classification
  var overlap: Float = DEFAULT_OVERLAP_VALUE,             // no clue
  var numOfResults: Int = DEFAULT_NUM_OF_RESULTS,         // number of output from tflite model
  var currentDelegate: Int = 0,                           // delegate processing to CPU or ???
  var numThreads: Int = 2                                 // no clue
) {
    private lateinit var classifier: AudioClassifier           // audioclassifier class - handles building/using tflite model
    private lateinit var tensorAudio: TensorAudio              // audio tensor(n-D array of data) used by tf
    private lateinit var recorder: AudioRecord                 // record from mic
    private lateinit var executor: ScheduledThreadPoolExecutor // executor - to execute ML over fixed intervals

    // call back function for running tflite model
    private val classifyRunnable = Runnable {
        classifyAudio() // executes this function
    }

    // function to run on init
    init {
        initClassifier()
    }

    // initialize/set up classifier
    fun initClassifier() {
        // Set general detection options, e.g. number of used threads
        val baseOptionsBuilder = BaseOptions.builder()
            .setNumThreads(numThreads)

        // Use the specified hardware for running the model. Default to CPU.
        // Possible to also use a GPU delegate, but this requires that the classifier be created
        // on the same thread that is using the classifier, which is outside of the scope of this
        // sample's design.
        when (currentDelegate) {
            DELEGATE_CPU -> {
                // Default
            }
            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        // Configures a set of parameters for the classifier and what results will be returned.
        // Model specific to yamnet - TODO: change for birdnet
        val options = AudioClassifier.AudioClassifierOptions.builder() // builder function for tflite model
            .setScoreThreshold(classificationThreshold) // threshold of acceptable error
            .setMaxResults(numOfResults)                // number of results returned
            .setBaseOptions(baseOptionsBuilder.build()) // no clue
            .build()                                    // start building

        try {
            // Create the classifier and required supporting objects
            classifier = AudioClassifier.createFromFileAndOptions(context,      // context to run in - android phone
                                                                  currentModel, // path to tflite model to run
                                                                  options)      // build options for model
            tensorAudio = classifier.createInputTensorAudio() // initializes tensor
            recorder = classifier.createAudioRecord()         // initializes audio recording
            startAudioClassification()                        // if successful build, start executing model
        } catch (e: IllegalStateException) {
            listener.onError(
                "Audio Classifier failed to initialize. See error logs for details"
            )

            Log.e("AudioClassification", "TFLite failed to load with error: " + e.message)
        }
    }

    // Starts executing/actually running classification from tflite model
    fun startAudioClassification() {
        // if still recording/mic in use, wait until finished
        if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
            return
        }

        recorder.startRecording()                            // get audio data from mic
        executor = ScheduledThreadPoolExecutor(1) // fires up tflite model to use

        // Each model will expect a specific audio recording length. This formula calculates that
        // length using the input buffer size and tensor format sample rate.
        // For example, YAMNET expects 0.975 second length recordings.
        // This needs to be in milliseconds to avoid the required Long value dropping decimals.
        val lengthInMilliSeconds = ((classifier.requiredInputBufferSize * 1.0f) /
                classifier.requiredTensorAudioFormat.sampleRate) * 1000

        val interval = (lengthInMilliSeconds * (1 - overlap)).toLong()

        // call/execute tflite model
        executor.scheduleAtFixedRate( // run model as a schedule at a fixed rate
            classifyRunnable,         // callback function to run audio classifier from tflite model
            0,              // no delay - start immediately
            interval,                 // record in specified interval/rate
            TimeUnit.MILLISECONDS)    // units of time
    }

    // what actually runs the tflite classifier
    private fun classifyAudio() {
        tensorAudio.load(recorder) // get audio recorded by mic - TODO: change to static audio file
        var inferenceTime = SystemClock.uptimeMillis()             // get time elapsed from boot
        val output = classifier.classify(tensorAudio)              // get output from classification
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime // calculate time took to run model
        listener.onResult(output[0].categories, inferenceTime)     // format result, and log how long it took
    }
    // stop/close app
    fun stopAudioClassification() {
        recorder.stop()        // stop recorder from mic
        executor.shutdownNow() // stop tflite model
    }

    companion object {
        const val DELEGATE_CPU = 0                                 // process on CPU
        const val DELEGATE_NNAPI = 1                               // use processing on NNAPI
        const val DISPLAY_THRESHOLD = 0.3f                         // acceptable error
        const val DEFAULT_NUM_OF_RESULTS = 2                       // default number of results to output
        const val DEFAULT_OVERLAP_VALUE = 0.5f                     // overlap in audio recording?
        const val YAMNET_MODEL = "yamnet.tflite"                   // classify type of audio
        const val SPEECH_COMMAND_MODEL = "speech.tflite"           // speech to text
        const val BIRDNET_MODEL = "BirdNET_6K_GLOBAL_MODEL.tflite" // birdnet tflite model
    }
}
