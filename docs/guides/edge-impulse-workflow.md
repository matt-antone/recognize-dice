# Edge Impulse Studio: Complete Dice Detection Workflow

## Prerequisites

- Edge Impulse account: https://studio.edgeimpulse.com
- Roboflow dice dataset downloaded and organized

## Step 1: Create New Project

1. Go to Edge Impulse Studio
2. Click "Create new project"
3. Name: "Pi3-Dice-Detection"
4. Project type: **"Images"**
5. Target device: **"Raspberry Pi 4"** ✅ (fully compatible with Pi 3 + works better with AI acceleration)

**Note:** Pi 4 target works perfectly on Pi 3 because:

- Same ARM64 architecture (binary compatible)
- IMX500 AI acceleration reduces CPU dependency
- TensorFlow Lite models are cross-compatible
- Pi 3 will actually perform similarly due to hardware AI acceleration

## Step 2: Data Acquisition

1. Click **"Data acquisition"** tab
2. Click **"Upload data"**
3. **Upload strategy:**
   - Upload all images as **"Automatically split between training and testing"**
   - Let Edge Impulse handle the 80/20 split
4. **Upload process:**
   - Select all dice images from your dataset
   - Choose **"Infer from filename"** or **"Enter label"**
   - Labels should be: `value_1`, `value_2`, `value_3`, `value_4`, `value_5`, `value_6`

## Step 3: Data Verification & Labeling

1. **Review uploaded data:**
   - Check that images are properly labeled
   - Verify training/test split looks good
   - Should see ~280 training images, ~79 test images
2. **Fix any labeling issues:**
   - Click on individual images to verify/correct labels
   - Ensure balanced distribution across classes

## Step 4: Impulse Design ⚙️

1. Click **"Impulse design"** tab
2. **Input block configuration:**
   - **Image width:** 160 pixels
   - **Image height:** 160 pixels
   - **Resize mode:** "Squash"
   - Click **"Add a processing block"**
3. **Processing block:**
   - Select **"Image"** block
   - This handles image preprocessing
4. **Learning block:**
   - Click **"Add a learning block"**
   - Select **"Transfer Learning (Images)"**
   - This uses MobileNet for dice classification
5. Click **"Save Impulse"**

## Step 5: Image Processing Configuration

1. Click **"Image"** in the left sidebar
2. **Parameters:**
   - **Color depth:** RGB (recommended for dice detection)
   - **Resize mode:** Squash (already set)
3. Click **"Save parameters"**
4. Click **"Generate features"**
   - Wait for processing to complete
   - This creates feature vectors from your images
5. **Review feature explorer:**
   - Should see good clustering of similar dice values
   - Different dice values should be well separated

## Step 6: Transfer Learning Configuration

1. Click **"Transfer learning"** in the left sidebar
2. **Model selection:**
   - **Pre-trained model:** MobileNetV2 160x160 1.0 (default)
   - **Output neurons:** Should auto-detect 6 classes
   - **Dropout rate:** 0.1 (prevents overfitting)
3. **Data augmentation:**
   - ✅ **Enabled** (recommended)
   - This creates variations of training images
4. **Training settings:**
   - **Training cycles:** 20 (start with this)
   - **Learning rate:** 0.005 (default)
   - **Batch size:** 32 (default)

## Step 7: Model Training 🚀

1. Click **"Start training"**
2. **Monitor progress:**
   - Watch accuracy and loss curves
   - Training should take 5-15 minutes
3. **Target metrics:**
   - **Accuracy:** >90% (aim for 95%+)
   - **Loss:** <0.3
   - **Inference time:** <100ms (important for Pi 3)

## Step 8: Model Testing

1. Click **"Model testing"** tab
2. Click **"Classify all"**
3. **Review results:**
   - Should see >90% accuracy on test set
   - Check for confusion between similar dice values
   - Note any misclassifications

## Step 9: Model Optimization (if needed)

If accuracy is low (<85%):

### Option A: Retrain with more cycles

1. Go back to Transfer learning
2. Increase training cycles to 30-50
3. Retrain

### Option B: Adjust model architecture

1. Try MobileNetV2 160x160 0.75 (faster, might be better for Pi 3)
2. Adjust dropout rate (0.2 for more regularization)

### Option C: Data augmentation tweaks

1. Increase augmentation if overfitting
2. Add more training data if underfitting

## Step 10: Deployment Preparation

1. Click **"Deployment"** tab
2. **Target platform:**
   - Search for **"Raspberry Pi"**
   - Select **"Raspberry Pi 4"**
3. **Processing target:**
   - Choose **"CPU"** ✅ (correct for Pi 3)
   - NOT "GPU" (Pi 3 GPU doesn't support ML acceleration)
4. **Optimization:**
   - ✅ **Enable EON compiler** (faster inference)
   - ✅ **Quantized (int8)** (smaller model size)

**Note:**

- **CPU target** is correct for all Raspberry Pi models
- **IMX500 AI acceleration** works alongside CPU inference
- **Pi 3 GPU** is for graphics only, not ML processing

## Step 11: Model Export

1. **Build method:**
   - Select **"TensorFlow Lite model"**
   - Alternative: **"Linux (ARM64)"** for complete package
2. Click **"Build"**
3. **Download:**
   - Model file: `model.tflite`
   - Labels file: `labels.txt`
   - Model metadata for integration

## Step 12: Integration with IMX500

1. **Transfer files to Pi 3:**

   ```bash
   scp model.tflite pi@<pi-ip>:~/dice_model.tflite
   scp labels.txt pi@<pi-ip>:~/dice_labels.txt
   ```

2. **Test model format compatibility:**
   ```bash
   # On Pi 3, verify model loads
   python3 -c "
   import tflite_runtime.interpreter as tflite
   interpreter = tflite.Interpreter('dice_model.tflite')
   print('Model loaded successfully!')
   "
   ```

## Troubleshooting Common Issues

### Issue: Poor accuracy (<80%)

**Solutions:**

- Check data labeling quality
- Increase training cycles
- Try different model architecture
- Add more training data

### Issue: Model too slow (>200ms on Pi 3)

**Solutions:**

- Use MobileNetV2 0.75 or 0.5 (faster)
- Reduce input image size to 96x96
- Enable int8 quantization

### Issue: Model file too large (>10MB)

**Solutions:**

- Enable quantization
- Use smaller MobileNet variant
- Prune model if available

### Issue: Deployment format not compatible

**Solutions:**

- Try different export format (ARM64 vs TFLite)
- Check TensorFlow Lite version compatibility
- Use RPK format if targeting IMX500 directly

## Expected Results

- **Accuracy:** 95%+ on dice values 1-6
- **Inference time:** 50-100ms on Pi 3
- **Model size:** 2-5MB (quantized)
- **Memory usage:** <100MB during inference

## Next Steps After Training

1. Test model on Pi 3 with static images
2. Integrate with picamera2 for live detection
3. Create real-time dice recognition application
4. Deploy to IMX500 for hardware acceleration
