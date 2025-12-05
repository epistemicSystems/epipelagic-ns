# Phase 2 Implementation Tasks

**Project**: Quran Memorization App - Phase 2 Enhancement
**Timeline**: Weeks 9-16 (8 weeks)
**Goal**: Improve accuracy to >95%, add Bret Victor visualizations, achieve 40% Day-7 retention
**Status**: Ready to implement
**Last Updated**: 2025-12-05

## Table of Contents
1. [Prerequisites Check](#prerequisites-check)
2. [Week 9: Data Collection & Analysis](#week-9-data-collection--analysis)
3. [Week 10-11: Whisper Fine-Tuning](#week-10-11-whisper-fine-tuning)
4. [Week 12: ML Error Classifier](#week-12-ml-error-classifier)
5. [Week 13-14: Bret Victor Visualizations](#week-13-14-bret-victor-visualizations)
6. [Week 15: Personalized Spaced Repetition](#week-15-personalized-spaced-repetition)
7. [Week 16: Iteration & Polish](#week-16-iteration--polish)
8. [Success Criteria](#success-criteria)
9. [Risk Mitigation](#risk-mitigation)

---

## Prerequisites Check

**Before starting Phase 2, verify Phase 1 (MVP) is complete:**

- [ ] **Backend API** (Node.js + Express)
  - [ ] User authentication (JWT) working
  - [ ] All API endpoints implemented (auth, verses, practice, reviews, progress)
  - [ ] PostgreSQL database with all tables
  - [ ] Redis caching configured
  - [ ] S3 audio storage working
  - [ ] Whisper ASR integration complete
  - [ ] Rule-based error detection implemented
  - [ ] SM-2 spaced repetition working

- [ ] **Frontend App** (React Native)
  - [ ] All 4 screens complete (Practice, Recall, Progress, Onboarding)
  - [ ] Audio recording/playback working
  - [ ] Redux state management configured
  - [ ] API integration complete

- [ ] **Infrastructure**
  - [ ] Docker Compose local dev environment
  - [ ] AWS staging environment deployed
  - [ ] Monitoring (Sentry) configured
  - [ ] CI/CD pipeline (GitHub Actions)

- [ ] **Metrics Baseline**
  - [ ] 100+ beta users active
  - [ ] ASR accuracy measured (current WER: __%)
  - [ ] Error detection accuracy measured (current: __%)
  - [ ] False positive rate measured (current: __%)
  - [ ] Day-7 retention measured (current: __%)

**Status**: ⚠️ If any prerequisites are incomplete, complete Phase 1 first before proceeding.

---

## Week 9: Data Collection & Analysis

**Goal**: Collect and analyze user data to inform Phase 2 improvements.

### 9.1: Backend Data Export Pipeline

**Priority**: P0 (Critical)
**Estimated Time**: 2 days
**Owner**: Backend Engineer

**Tasks**:
- [ ] Create `/admin/export/recitations` endpoint
  - [ ] Authenticate admin users only
  - [ ] Support date range filters
  - [ ] Export fields: user_id, verse_id, audio_s3_key, transcription, asr_confidence, errors, accuracy, user_feedback
  - [ ] Stream large datasets (don't load all in memory)
  - [ ] Export to CSV/JSON format

- [ ] Create data export script
  - [ ] Script: `backend/scripts/export-recitations.ts`
  - [ ] Download audio files from S3
  - [ ] Organize by: recitations/{user_id}/{verse_id}/{timestamp}.wav
  - [ ] Create manifest.json with metadata
  - [ ] Compress to .tar.gz

- [ ] Testing
  - [ ] Test with 1000+ recitations
  - [ ] Verify data integrity
  - [ ] Document export process

**Acceptance Criteria**:
- Export 1000+ recitations with metadata
- Audio files downloadable and playable
- CSV includes all required fields

---

### 9.2: ASR Accuracy Analysis

**Priority**: P0 (Critical)
**Estimated Time**: 3 days
**Owner**: ML Engineer

**Tasks**:
- [ ] Manual transcription labeling
  - [ ] Sample 100 recitations (stratified by confidence score)
  - [ ] Hire 2 Arabic speakers for ground truth labeling
  - [ ] Create labeling guidelines document
  - [ ] Use Label Studio or similar tool
  - [ ] Store labels in `ml/data/ground_truth_labels.json`

- [ ] Calculate accuracy metrics
  - [ ] Script: `ml/scripts/analyze_asr_accuracy.py`
  - [ ] Compute Word Error Rate (WER) = (S + D + I) / N
    - S = Substitutions, D = Deletions, I = Insertions, N = Total words
  - [ ] Break down by:
    - Confidence score ranges (0-0.6, 0.6-0.8, 0.8-1.0)
    - Verse length (short <10 words, medium 10-20, long >20)
    - User experience level (new, intermediate, advanced)
  - [ ] Compare to baseline Whisper

- [ ] Error pattern analysis
  - [ ] Identify most common substitution errors
  - [ ] Analyze phonetically similar mistakes
  - [ ] Document dialect variations
  - [ ] Create error taxonomy

**Acceptance Criteria**:
- WER calculated on 100+ labeled samples
- Report showing accuracy breakdown by category
- Top 20 error patterns identified

---

### 9.3: Error Detection Analysis

**Priority**: P0 (Critical)
**Estimated Time**: 2 days
**Owner**: ML Engineer

**Tasks**:
- [ ] Analyze user feedback
  - [ ] Query all "incorrect detection" flags from DB
  - [ ] Calculate false positive rate: FP / (FP + TN)
  - [ ] Identify patterns in false positives
  - [ ] Document edge cases

- [ ] Precision/Recall analysis
  - [ ] Create confusion matrix
  - [ ] Calculate:
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
  - [ ] Compare to target: >85% accuracy, <15% FPR

- [ ] Report generation
  - [ ] Script: `ml/scripts/analyze_error_detection.py`
  - [ ] Generate PDF report with charts
  - [ ] Share with team for review

**Acceptance Criteria**:
- False positive rate calculated
- Report showing top failure cases
- Action items for improvement

---

### 9.4: User Feedback Analysis

**Priority**: P1 (Important)
**Estimated Time**: 2 days
**Owner**: Product/Frontend Engineer

**Tasks**:
- [ ] User interviews
  - [ ] Schedule 10 interviews with active users
  - [ ] Prepare interview script
  - [ ] Focus areas:
    - Error detection trust
    - Spaced repetition experience
    - Feature requests
    - Pain points
  - [ ] Record and transcribe interviews

- [ ] Quantitative analysis
  - [ ] Analyze app analytics (screen time, button clicks)
  - [ ] Identify drop-off points
  - [ ] Calculate feature usage rates
  - [ ] Retention cohorts (Day-1, Day-7, Day-30)

- [ ] Prioritization
  - [ ] Create improvement backlog
  - [ ] Prioritize using RICE framework:
    - Reach * Impact * Confidence / Effort
  - [ ] Document in TASKS.md

**Acceptance Criteria**:
- 10 user interviews completed
- Analytics report generated
- Prioritized improvement list

---

### Week 9 Deliverables
- [ ] 1000+ recitations exported and labeled
- [ ] ASR accuracy report (WER, error patterns)
- [ ] Error detection analysis (FPR, precision/recall)
- [ ] User feedback summary
- [ ] Phase 2 priorities finalized

---

## Week 10-11: Whisper Fine-Tuning

**Goal**: Fine-tune Whisper model on Quranic recitations to reduce WER from ~15% to <8%.

### 10.1: Dataset Acquisition

**Priority**: P0 (Critical)
**Estimated Time**: 3 days
**Owner**: ML Engineer

**Tasks**:
- [ ] Research existing datasets
  - [ ] Check Tarteel.ai open dataset
  - [ ] Check Quran Audio dataset (everyayah.com)
  - [ ] Contact Islamic organizations for data
  - [ ] License verification

- [ ] Data collection options
  - **Option A**: Use existing public dataset
    - [ ] Download Tarteel dataset (~500 hours)
    - [ ] Verify quality and licensing
  - **Option B**: Collect from users
    - [ ] Incentivize users to donate recordings
    - [ ] Ensure informed consent
  - **Option C**: Hire reciters
    - [ ] Budget: $2000-5000 for 100 hours
    - [ ] Find professional Qaris

- [ ] Dataset preparation
  - [ ] Format: Audio files + transcription JSON
  - [ ] Directory structure:
    ```
    ml/data/fine_tuning/
    ├── audio/
    │   ├── 001_001_001.wav
    │   ├── 001_002_001.wav
    │   └── ...
    ├── transcriptions.json
    └── metadata.json
    ```
  - [ ] Validate all files (16kHz, mono, WAV format)
  - [ ] Total target: 100-500 hours

**Acceptance Criteria**:
- 100+ hours of Quranic audio with transcriptions
- All files validated and formatted correctly
- Dataset split ready (train/val/test)

---

### 10.2: Fine-Tuning Pipeline Setup

**Priority**: P0 (Critical)
**Estimated Time**: 4 days
**Owner**: ML Engineer

**Tasks**:
- [ ] Environment setup
  - [ ] GPU instance: AWS p3.2xlarge or equivalent
  - [ ] Install dependencies:
    - PyTorch 2.0+
    - Transformers (Hugging Face)
    - Datasets library
    - Whisper model weights
  - [ ] Configure CUDA/cuDNN

- [ ] Data preprocessing
  - [ ] Script: `ml/src/training/prepare_dataset.py`
  - [ ] Convert audio to 16kHz mono
  - [ ] Normalize audio levels
  - [ ] Create Hugging Face Dataset format
  - [ ] Split: 80% train, 10% val, 10% test
  - [ ] Data augmentation (optional):
    - Speed perturbation (0.9x - 1.1x)
    - Background noise injection
    - Pitch shift

- [ ] Fine-tuning script
  - [ ] Script: `ml/src/training/finetune_whisper.py`
  - [ ] Base model: `openai/whisper-medium` or `whisper-large-v3`
  - [ ] Hyperparameters:
    ```python
    learning_rate = 1e-5
    batch_size = 16  # Adjust based on GPU memory
    epochs = 5-10
    warmup_steps = 500
    gradient_accumulation_steps = 2
    fp16 = True  # Mixed precision training
    ```
  - [ ] Use LoRA (Low-Rank Adaptation) for efficiency
  - [ ] Implement early stopping (patience=3)
  - [ ] Save checkpoints every epoch

**Acceptance Criteria**:
- Training pipeline runs without errors
- Model checkpoints saved
- Training logs available

---

### 10.3: Model Training & Evaluation

**Priority**: P0 (Critical)
**Estimated Time**: 5 days (includes compute time)
**Owner**: ML Engineer

**Tasks**:
- [ ] Train fine-tuned model
  - [ ] Run training script
  - [ ] Monitor metrics:
    - Training loss
    - Validation WER
    - GPU utilization
  - [ ] Expected training time: 12-24 hours (100 hours of audio)
  - [ ] Use TensorBoard for visualization

- [ ] Model evaluation
  - [ ] Script: `ml/src/training/evaluate_whisper.py`
  - [ ] Compute WER on test set
  - [ ] Compare to baseline Whisper
  - [ ] Break down by:
    - Verse length
    - Speaker characteristics
    - Audio quality
  - [ ] Target: WER < 8%

- [ ] Model optimization
  - [ ] Quantize to INT8 (optional, for speed)
  - [ ] Convert to ONNX format (optional)
  - [ ] Test inference speed
  - [ ] Target latency: <1.5s (p95)

**Acceptance Criteria**:
- Fine-tuned model achieves WER < 8% on test set
- Model size reasonable (<3GB)
- Inference latency acceptable

---

### 10.4: Model Deployment

**Priority**: P0 (Critical)
**Estimated Time**: 2 days
**Owner**: ML + Backend Engineer

**Tasks**:
- [ ] Model hosting options
  - **Option A**: Self-hosted FastAPI
    - [ ] Update `ml/src/inference/asr_service.py`
    - [ ] Load fine-tuned model
    - [ ] Deploy to AWS EC2 (g4dn.xlarge GPU instance)
  - **Option B**: AWS SageMaker
    - [ ] Create SageMaker endpoint
    - [ ] Deploy model
    - [ ] Set up auto-scaling
  - **Option C**: OpenAI fine-tuned API (if supported)

- [ ] API integration
  - [ ] Update `backend/src/services/asrService.ts`
  - [ ] Add model version selector (baseline vs fine-tuned)
  - [ ] Implement A/B testing (50% baseline, 50% fine-tuned)
  - [ ] Log all results for comparison

- [ ] Testing
  - [ ] Integration tests
  - [ ] Load testing (100 concurrent requests)
  - [ ] Monitor cost per transcription

**Acceptance Criteria**:
- Fine-tuned model deployed and accessible via API
- A/B testing active
- Monitoring in place

---

### Week 10-11 Deliverables
- [ ] Fine-tuned Whisper model (WER < 8%)
- [ ] Model deployed to production (A/B testing)
- [ ] Evaluation report comparing baseline vs fine-tuned
- [ ] Cost analysis and projections

---

## Week 12: ML Error Classifier

**Goal**: Build ML-based error classifier to reduce false positive rate from 15% to <5%.

### 12.1: Feature Engineering

**Priority**: P0 (Critical)
**Estimated Time**: 2 days
**Owner**: ML Engineer

**Tasks**:
- [ ] Define features
  - [ ] ASR confidence score (0-1)
  - [ ] Levenshtein distance (normalized)
  - [ ] Phonetic similarity (RapidFuzz)
  - [ ] Word count difference
  - [ ] Character count difference
  - [ ] Verse difficulty level
  - [ ] User experience level (total verses memorized)
  - [ ] Average accuracy for this verse (historical)
  - [ ] Time since last practice

- [ ] Feature extraction script
  - [ ] Script: `ml/src/error_classifier/extract_features.py`
  - [ ] Process all labeled data from Week 9
  - [ ] Create feature matrix (CSV)
  - [ ] Handle missing values
  - [ ] Feature normalization (StandardScaler)

- [ ] Feature analysis
  - [ ] Correlation matrix
  - [ ] Feature importance (Random Forest)
  - [ ] Remove redundant features

**Acceptance Criteria**:
- Feature matrix created with 1000+ samples
- Top 10 most important features identified
- Feature engineering pipeline documented

---

### 12.2: Model Training

**Priority**: P0 (Critical)
**Estimated Time**: 3 days
**Owner**: ML Engineer

**Tasks**:
- [ ] Prepare training data
  - [ ] Labels: 0 (no error) vs 1 (error detected)
  - [ ] Handle class imbalance (SMOTE or class weights)
  - [ ] Split: 80% train, 20% test
  - [ ] Stratified sampling

- [ ] Train multiple models
  - [ ] Logistic Regression (baseline)
  - [ ] Random Forest
  - [ ] Gradient Boosting (XGBoost)
  - [ ] Neural Network (simple MLP)
  - [ ] Script: `ml/src/error_classifier/train_classifier.py`

- [ ] Hyperparameter tuning
  - [ ] Use GridSearchCV or Optuna
  - [ ] 5-fold cross-validation
  - [ ] Optimize for F1 score (balance precision/recall)
  - [ ] Tune threshold (default 0.5 may not be optimal)

- [ ] Model evaluation
  - [ ] Test set metrics:
    - Precision (minimize false positives)
    - Recall (don't miss real errors)
    - F1 Score
    - ROC-AUC
  - [ ] Confusion matrix
  - [ ] Compare all models
  - [ ] Select best model (target: FPR < 5%)

**Acceptance Criteria**:
- Trained classifier achieves FPR < 5%, Recall > 85%
- Model comparison report
- Best model saved as `ml/models/error_classifier_v1.pkl`

---

### 12.3: Production Integration

**Priority**: P0 (Critical)
**Estimated Time**: 2 days
**Owner**: ML + Backend Engineer

**Tasks**:
- [ ] Model deployment
  - [ ] Create FastAPI endpoint: `/ml/classify-errors`
  - [ ] Load trained model
  - [ ] Implement feature extraction in production
  - [ ] Script: `ml/src/inference/error_classifier_service.py`

- [ ] Backend integration
  - [ ] Update `backend/src/services/errorClassifierService.ts`
  - [ ] Call ML service for classification
  - [ ] Fallback to rule-based if ML fails
  - [ ] A/B testing: 50% rule-based, 50% ML

- [ ] Monitoring
  - [ ] Log all predictions
  - [ ] Track false positive rate (user feedback)
  - [ ] Alert if FPR > 8%
  - [ ] Retrain trigger (weekly)

**Acceptance Criteria**:
- ML classifier deployed to production
- A/B testing active
- Monitoring dashboard showing FPR

---

### Week 12 Deliverables
- [ ] ML error classifier (FPR < 5%)
- [ ] Deployed to production with A/B testing
- [ ] Monitoring and alerting configured
- [ ] Model retraining pipeline documented

---

## Week 13-14: Bret Victor Visualizations

**Goal**: Implement 3 interactive visualizations to make learning "understandable".

### 13.1: Memory Strength Visualizer

**Priority**: P0 (Critical)
**Estimated Time**: 4 days
**Owner**: Frontend Engineer

**Tasks**:
- [ ] Backend API
  - [ ] Endpoint: `GET /progress/memory-curve/:verseId`
  - [ ] Calculate memory strength over time:
    ```typescript
    memoryStrength(t) = initialStrength * e^(-t / easeFactor)
    ```
  - [ ] Return time series data:
    ```json
    {
      "verseId": 1,
      "currentStrength": 0.85,
      "history": [
        { "date": "2025-11-01", "strength": 1.0, "event": "memorized" },
        { "date": "2025-11-02", "strength": 0.95, "event": "review" },
        { "date": "2025-11-09", "strength": 0.85, "event": "review" }
      ],
      "nextReview": "2025-11-16",
      "forgettingCurve": [
        { "daysFromNow": 0, "predictedStrength": 0.85 },
        { "daysFromNow": 7, "predictedStrength": 0.70 },
        { "daysFromNow": 14, "predictedStrength": 0.50 }
      ]
    }
    ```

- [ ] Frontend component
  - [ ] Component: `frontend/src/components/MemoryVisualizer.tsx`
  - [ ] Use Recharts or Victory Native
  - [ ] Show:
    - Line chart: Memory strength over time
    - Dotted line: Predicted forgetting curve
    - Markers: Review events
    - Interactive: Tap to see details
  - [ ] Color coding:
    - Green: Strength > 80%
    - Yellow: 50-80%
    - Red: < 50%

- [ ] Interactivity
  - [ ] Tap on date to see review details
  - [ ] Swipe to explore different verses
  - [ ] "What if" mode: Adjust review timing, see predicted impact
  - [ ] Tooltip showing next review date

**Acceptance Criteria**:
- Memory curve displays for any verse
- Interactive and smooth (60fps)
- Users understand their memory state

---

### 13.2: Pronunciation Explorer (Waveform Comparison)

**Priority**: P1 (Important)
**Estimated Time**: 5 days
**Owner**: Frontend + ML Engineer

**Tasks**:
- [ ] Backend processing
  - [ ] Endpoint: `GET /practice/pronunciation-analysis/:recitationId`
  - [ ] Load user recording from S3
  - [ ] Load reference Qari recording
  - [ ] Extract waveforms (using librosa)
  - [ ] Align audio (DTW - Dynamic Time Warping)
  - [ ] Return:
    ```json
    {
      "userWaveform": [...],  // Amplitude values
      "qariWaveform": [...],
      "alignment": [
        { "userTime": 0.5, "qariTime": 0.45, "similarity": 0.92 }
      ],
      "phonemeTimestamps": [
        { "phoneme": "ب", "userTime": 0.1, "qariTime": 0.09 }
      ]
    }
    ```

- [ ] Frontend component
  - [ ] Component: `frontend/src/components/PronunciationExplorer.tsx`
  - [ ] Dual waveform display (user vs Qari)
  - [ ] Synchronized playback
  - [ ] Click on waveform to play that section
  - [ ] Highlight differences (low similarity areas)
  - [ ] Phoneme-level markers

- [ ] Advanced features (optional)
  - [ ] Spectrogram view (frequency analysis)
  - [ ] Playback speed control (0.5x - 1.5x)
  - [ ] Loop problematic sections
  - [ ] Export comparison video

**Acceptance Criteria**:
- Waveform comparison displays correctly
- User can identify pronunciation differences
- Performance acceptable (renders < 1s)

---

### 13.3: Spaced Repetition Explorer

**Priority**: P1 (Important)
**Estimated Time**: 4 days
**Owner**: Frontend + Backend Engineer

**Tasks**:
- [ ] Backend simulation API
  - [ ] Endpoint: `POST /reviews/simulate`
  - [ ] Request:
    ```json
    {
      "verseId": 1,
      "scenario": {
        "easeFactor": 2.5,
        "qualityRatings": [4, 4, 5, 3]  // Future ratings
      }
    }
    ```
  - [ ] Simulate SM-2 algorithm
  - [ ] Return predicted schedule:
    ```json
    {
      "reviews": [
        { "date": "2025-11-10", "interval": 7 },
        { "date": "2025-11-17", "interval": 14 },
        { "date": "2025-12-01", "interval": 28 }
      ],
      "predictedRetention": 0.85
    }
    ```

- [ ] Frontend component
  - [ ] Component: `frontend/src/components/SpacedRepExplorer.tsx`
  - [ ] Interactive sliders:
    - Ease factor (1.3 - 3.0)
    - Quality ratings (0-5)
  - [ ] Real-time chart update
  - [ ] Show:
    - Review timeline
    - Predicted retention curve
    - Total reviews needed
  - [ ] Compare strategies:
    - Standard SM-2
    - Aggressive (shorter intervals)
    - Conservative (longer intervals)

- [ ] Educational content
  - [ ] Explain SM-2 algorithm in simple terms
  - [ ] Tooltip: "Ease factor = how easy this verse is for you"
  - [ ] Show impact of ratings on schedule

**Acceptance Criteria**:
- Users can explore different scheduling strategies
- Real-time feedback (<100ms latency)
- Educational and intuitive

---

### Week 13-14 Deliverables
- [ ] Memory Strength Visualizer (live)
- [ ] Pronunciation Explorer (live)
- [ ] Spaced Repetition Explorer (live)
- [ ] User testing with 10 beta users
- [ ] Documentation and tutorial videos

---

## Week 15: Personalized Spaced Repetition

**Goal**: Adapt SM-2 intervals based on individual user performance.

### 15.1: User Performance Analysis

**Priority**: P0 (Critical)
**Estimated Time**: 2 days
**Owner**: ML Engineer

**Tasks**:
- [ ] Data collection
  - [ ] Query all review events per user
  - [ ] Calculate per-user metrics:
    - Average accuracy
    - Review completion rate
    - Typical quality ratings
    - Retention rate (% remembered after N days)
  - [ ] Group users into cohorts:
    - Fast learners (high accuracy, low repetitions)
    - Average learners
    - Struggling learners (low accuracy, many repetitions)

- [ ] Retention modeling
  - [ ] Script: `ml/src/spaced_repetition/analyze_retention.py`
  - [ ] For each user, fit retention curve:
    ```python
    retention(t) = e^(-t / halfLife)
    # halfLife varies per user
    ```
  - [ ] Estimate optimal intervals per user
  - [ ] Identify over-scheduled vs under-scheduled users

**Acceptance Criteria**:
- Retention curves calculated for 100+ users
- User cohorts identified
- Optimal intervals estimated

---

### 15.2: Adaptive SM-2 Algorithm

**Priority**: P0 (Critical)
**Estimated Time**: 3 days
**Owner**: ML Engineer

**Tasks**:
- [ ] Personalization model
  - [ ] Script: `ml/src/spaced_repetition/personalized_sm2.py`
  - [ ] Inputs:
    - User's historical performance
    - Current ease factor
    - Verse difficulty
    - Recent accuracy trend
  - [ ] Adjustments:
    - Shorten intervals if accuracy declining
    - Lengthen intervals if consistently high accuracy
    - Adjust ease factor based on user's retention rate
  - [ ] Algorithm:
    ```python
    def personalized_interval(
        base_interval: int,
        user_half_life: float,
        global_half_life: float = 14.0
    ) -> int:
        # Adjust interval based on user's learning speed
        adjustment_factor = user_half_life / global_half_life
        return round(base_interval * adjustment_factor)
    ```

- [ ] Implementation
  - [ ] Update `backend/src/services/spacedRepetitionService.ts`
  - [ ] Call ML service for personalized intervals
  - [ ] Fallback to standard SM-2 for new users (< 10 reviews)
  - [ ] Log all interval calculations

- [ ] A/B testing setup
  - [ ] Group A: Standard SM-2
  - [ ] Group B: Personalized SM-2
  - [ ] Randomly assign users (50/50)
  - [ ] Track metrics: retention, completion rate, user satisfaction

**Acceptance Criteria**:
- Personalized SM-2 implemented
- A/B test running
- Monitoring dashboard active

---

### 15.3: User Control & Transparency

**Priority**: P1 (Important)
**Estimated Time**: 2 days
**Owner**: Frontend Engineer

**Tasks**:
- [ ] Settings screen update
  - [ ] Screen: `frontend/src/screens/SettingsScreen.tsx`
  - [ ] Add "Learning Pace" section:
    - [ ] Slider: Slower ←→ Faster
    - [ ] Options: Conservative, Standard, Aggressive
    - [ ] Explain impact on intervals
  - [ ] Show personalized stats:
    - "Your average retention: 85%"
    - "Optimal review frequency: Every 10 days"

- [ ] Transparency features
  - [ ] On review screen, show:
    - "Next review in 12 days (personalized for you)"
    - "Standard schedule would be: 14 days"
  - [ ] Allow override:
    - "Review sooner" button
    - "Skip this review" (if very confident)

**Acceptance Criteria**:
- Users can control their learning pace
- Personalization is transparent and understandable
- Override options work correctly

---

### Week 15 Deliverables
- [ ] Personalized SM-2 algorithm deployed
- [ ] A/B testing active (50% control, 50% personalized)
- [ ] User controls and transparency features live
- [ ] Monitoring retention improvements

---

## Week 16: Iteration & Polish

**Goal**: Optimize performance, fix bugs, polish UX based on feedback.

### 16.1: Performance Optimization

**Priority**: P0 (Critical)
**Estimated Time**: 3 days
**Owner**: Full Stack Team

**Tasks**:
- [ ] Backend optimization
  - [ ] Identify slow queries (EXPLAIN ANALYZE)
  - [ ] Add missing database indexes
  - [ ] Optimize N+1 queries
  - [ ] Implement query caching (Redis)
  - [ ] Target: p95 latency < 500ms for all endpoints

- [ ] Frontend optimization
  - [ ] Reduce bundle size:
    - [ ] Code splitting (React.lazy)
    - [ ] Remove unused dependencies
    - [ ] Tree shaking
  - [ ] Optimize re-renders:
    - [ ] Use React.memo for expensive components
    - [ ] useMemo for heavy computations
    - [ ] Avoid inline functions in props
  - [ ] Image optimization:
    - [ ] Compress images
    - [ ] Use WebP format
    - [ ] Lazy load images

- [ ] ML service optimization
  - [ ] Model quantization (INT8) if not done
  - [ ] Batch inference for multiple requests
  - [ ] GPU utilization monitoring
  - [ ] Auto-scaling based on load

**Acceptance Criteria**:
- API p95 latency < 500ms
- App launch time < 3s
- 60fps maintained during animations

---

### 16.2: Bug Fixes & Edge Cases

**Priority**: P0 (Critical)
**Estimated Time**: 2 days
**Owner**: Full Stack Team

**Tasks**:
- [ ] Review bug reports from beta users
  - [ ] Prioritize: P0 (critical), P1 (important), P2 (nice to have)
  - [ ] Fix all P0 bugs
  - [ ] Fix top 10 P1 bugs

- [ ] Edge case handling
  - [ ] Handle offline mode gracefully
  - [ ] Handle low disk space
  - [ ] Handle audio permission denied
  - [ ] Handle API timeouts
  - [ ] Handle invalid JWT tokens
  - [ ] Handle concurrent requests

- [ ] Error message improvements
  - [ ] Replace generic errors with specific, helpful messages
  - [ ] Add retry buttons where appropriate
  - [ ] Provide contact support option

**Acceptance Criteria**:
- All P0 bugs fixed
- Edge cases handled gracefully
- Error messages user-friendly

---

### 16.3: UX Polish

**Priority**: P1 (Important)
**Estimated Time**: 2 days
**Owner**: Frontend Engineer

**Tasks**:
- [ ] Onboarding improvements
  - [ ] Simplify to 3 steps maximum
  - [ ] Add skip option
  - [ ] Show value proposition clearly
  - [ ] A/B test different flows

- [ ] Microinteractions
  - [ ] Loading skeletons (instead of spinners)
  - [ ] Smooth transitions between screens
  - [ ] Haptic feedback on important actions
  - [ ] Success animations (celebrate achievements)

- [ ] Accessibility
  - [ ] Screen reader support
  - [ ] Color contrast (WCAG AA)
  - [ ] Font size adjustable
  - [ ] Tap targets ≥ 44x44pt

- [ ] Empty states
  - [ ] Meaningful illustrations
  - [ ] Clear calls to action
  - [ ] Helpful hints

**Acceptance Criteria**:
- Onboarding completion rate > 80%
- App feels smooth and polished
- Accessibility audit passing

---

### 16.4: Documentation & Testing

**Priority**: P1 (Important)
**Estimated Time**: 2 days
**Owner**: Full Stack Team

**Tasks**:
- [ ] Update documentation
  - [ ] Update CLAUDE.md with Phase 2 features
  - [ ] Update API_REFERENCE.md
  - [ ] Update ARCHITECTURE.md (new ML services)
  - [ ] Create PHASE2_REPORT.md (results, learnings)

- [ ] Testing coverage
  - [ ] Backend unit tests > 80%
  - [ ] Frontend component tests > 70%
  - [ ] ML model tests (accuracy, edge cases)
  - [ ] E2E tests for critical flows

- [ ] Load testing
  - [ ] Simulate 1000 concurrent users
  - [ ] Monitor error rates, latency
  - [ ] Identify bottlenecks
  - [ ] Fix scalability issues

**Acceptance Criteria**:
- Documentation up to date
- Test coverage meets targets
- Load testing successful (no errors at 1000 users)

---

### Week 16 Deliverables
- [ ] All performance optimizations complete
- [ ] All P0 bugs fixed
- [ ] UX polished based on feedback
- [ ] Documentation updated
- [ ] Phase 2 ready for launch

---

## Success Criteria

### Technical Metrics

| Metric | Phase 1 Baseline | Phase 2 Target | Current |
|--------|------------------|----------------|---------|
| ASR Accuracy (WER) | ~15% | < 8% | - |
| Error Detection FPR | ~15% | < 5% | - |
| Error Detection Recall | ~85% | > 90% | - |
| API p95 Latency | ~800ms | < 500ms | - |
| App Bundle Size | ~20MB | < 15MB | - |
| Memory Strength Viz | N/A | Live | - |
| Pronunciation Explorer | N/A | Live | - |
| Spaced Rep Explorer | N/A | Live | - |
| Personalized SM-2 | N/A | A/B testing | - |

### User Metrics

| Metric | Phase 1 Baseline | Phase 2 Target | Current |
|--------|------------------|----------------|---------|
| Day-7 Retention | ~30% | > 40% | - |
| Day-30 Retention | ~15% | > 20% | - |
| User Satisfaction (NPS) | ~40 | > 50 | - |
| Error Detection Trust | ~3.5/5 | > 4/5 | - |
| Feature Usage (Visualizations) | 0% | > 30% | - |
| Review Completion Rate | ~50% | > 60% | - |

### Business Metrics

| Metric | Phase 1 Baseline | Phase 2 Target | Current |
|--------|------------------|----------------|---------|
| Total Active Users | ~100 | 500-1000 | - |
| ASR Cost per Recitation | ~$0.003 | < $0.002 | - |
| Infrastructure Cost/Month | ~$200 | < $500 | - |
| Bug Reports per 100 Users | ~15 | < 5 | - |

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| Fine-tuning doesn't improve WER | Medium | High | Use multiple architectures (Whisper Large, wav2vec2); collect more data; consider ensemble |
| ML classifier overfits | Medium | Medium | Cross-validation; regularization; collect more diverse data |
| Visualizations too slow | Low | Medium | Optimize rendering (canvas vs SVG); lazy loading; reduce data points |
| GPU costs exceed budget | Medium | Medium | Batch inference; model quantization; consider serverless GPU |
| Database becomes bottleneck | Low | High | Add read replicas; optimize queries; implement caching |

### Product Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| Users don't understand visualizations | High | Medium | User testing; simplify; add tutorials; make optional |
| Personalization doesn't improve retention | Medium | High | A/B testing; allow manual override; iterate based on data |
| Phase 2 features not compelling | Low | High | User interviews; prioritize most requested features; iterate quickly |
| Technical debt accumulates | Medium | Medium | Allocate 20% time for refactoring; code reviews; documentation |

### Schedule Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| Fine-tuning takes longer than expected | High | Medium | Start early; have backup (use baseline Whisper); parallelproduct work |
| Dataset acquisition delayed | Medium | High | Multiple sources; start outreach early; budget for paid data |
| Scope creep | High | Medium | Strict prioritization; say no to low-impact features; defer to Phase 3 |
| Key team member unavailable | Low | High | Cross-training; documentation; clear handoff procedures |

---

## Implementation Order (Critical Path)

**Parallel Work Streams:**

1. **ML Stream** (ML Engineer):
   - Week 9: Data collection & analysis
   - Week 10-11: Whisper fine-tuning (CRITICAL PATH)
   - Week 12: ML error classifier
   - Week 15: Personalized SM-2

2. **Frontend Stream** (Frontend Engineer):
   - Week 9-10: Memory Visualizer
   - Week 11-12: Pronunciation Explorer
   - Week 13: Spaced Rep Explorer
   - Week 14-15: Settings & user controls
   - Week 16: Polish & optimization

3. **Backend Stream** (Backend Engineer):
   - Week 9: Data export APIs
   - Week 10-11: Model deployment infrastructure
   - Week 12: ML classifier integration
   - Week 13-14: Visualization APIs
   - Week 15: Personalized SM-2 APIs
   - Week 16: Performance optimization

**Dependencies:**
- Visualizations depend on data collection (Week 9)
- ML classifier depends on labeled data (Week 9)
- Personalized SM-2 depends on user performance data (Week 9)
- Fine-tuning is independent and should start ASAP

---

## Daily Workflow

**Daily Standup** (15 min, async on Slack):
- What did I complete yesterday?
- What am I working on today?
- Any blockers?

**Weekly Sprint** (Friday):
- Demo completed work
- Review metrics
- Plan next week
- Update TASKS.md

**Communication:**
- Urgent issues: Slack DM
- Questions: Slack #engineering channel
- Code reviews: GitHub PR
- Design discussions: Figma comments

---

## Getting Started

**Step 1**: Verify Prerequisites
```bash
# Check Phase 1 completion
make check-phase1-complete

# Expected output: All checks ✅
```

**Step 2**: Set up Week 9 environment
```bash
# Install ML dependencies
cd ml
pip install -r requirements-phase2.txt

# Set up data directories
mkdir -p data/raw data/processed data/ground_truth
```

**Step 3**: Start with data collection
```bash
# Export recitations for analysis
npm run script:export-recitations -- --limit 1000 --output data/raw/

# Begin manual labeling
python ml/scripts/setup_labeling_tool.py
```

**Step 4**: Daily updates
```bash
# Update task status
# Edit TASKS.md, mark completed tasks with [x]

# Commit progress
git add TASKS.md
git commit -m "docs: Update Phase 2 progress - Week 9 Day 1"
git push
```

---

**Questions?** Contact the team lead or review @CLAUDE.md for project guidelines.

**Ready to start?** Begin with Week 9, Task 9.1: Backend Data Export Pipeline.
