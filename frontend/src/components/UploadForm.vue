<template>
  <div class="upload-form card fade-in-up">
    <div class="form-header">
      <h1 class="form-title" style="text-align: center;">Upload Video</h1>
    </div>

    <!-- Drop zone -->
    <div
      class="drop-zone"
      :class="{ 'drop-zone--active': isDragging, 'drop-zone--has-file': selectedFile }"
      @dragover.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @drop.prevent="onDrop"
      @click="$refs.fileInput.click()"
    >
      <input
        ref="fileInput"
        type="file"
        accept="video/*"
        class="file-input-hidden"
        @change="onFileSelect"
      />

      <div v-if="!selectedFile" class="drop-zone__placeholder">
        <div class="drop-zone__icon">
          <img src="../assets/file-up.svg" style="width: 44px; height: 44px;" alt="File Up" />
        </div>
        <p class="drop-zone__text">Drag &amp; drop your video, or click to browse.</p>
        <ol class="drop-zone__rules">
          <li>Video must be under 5 mins.</li>
          <li>Valid file formats: .mp4, .mkv.</li>
        </ol>
      </div>

      <div v-else class="drop-zone__preview">
        <div class="file-badge">
          <img src="../assets/video.svg" style="width: 32px; height: 32px;" alt="Video" />
          <span class="file-name">{{ selectedFile.name }}</span>
          <span class="file-size">{{ formatSize(selectedFile.size) }}</span>
        </div>
        <button class="remove-btn" @click.stop="removeFile">✕</button>
      </div>
    </div>

    <!-- Email input -->
    <div class="email-group">
      <label class="input-label" for="email">Email</label>
      <input
        id="email"
        v-model="email"
        type="email"
        class="input-field"
        :disabled="uploading"
      />
    </div>

    <!-- Submit -->
    <button
      class="btn btn-primary btn-submit"
      :disabled="!canSubmit"
      @click="submit"
    >
      <span v-if="uploading" class="spinner"></span>
      {{ uploading ? 'Uploading…' : 'Upload & Analysis' }}
    </button>

    <!-- Error message -->
    <p v-if="errorMsg" class="error-msg">{{ errorMsg }}</p>
  </div>
</template>

<script>
export default {
  name: 'UploadForm',
  emits: ['job-started'],
  data() {
    return {
      selectedFile: null,
      email: '',
      isDragging: false,
      uploading: false,
      errorMsg: '',
    }
  },
  computed: {
    canSubmit() {
      return this.selectedFile && this.email.trim() && !this.uploading
    },
  },
  methods: {
    onDrop(e) {
      this.isDragging = false
      const file = e.dataTransfer.files[0]
      if (file && file.type.startsWith('video/')) {
        this.selectedFile = file
        this.errorMsg = ''
      } else {
        this.errorMsg = 'Please drop a valid video file.'
      }
    },
    onFileSelect(e) {
      const file = e.target.files[0]
      if (file) {
        this.selectedFile = file
        this.errorMsg = ''
      }
    },
    removeFile() {
      this.selectedFile = null
      this.$refs.fileInput.value = ''
    },
    formatSize(bytes) {
      if (bytes < 1024) return bytes + ' B'
      if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB'
      return (bytes / 1048576).toFixed(1) + ' MB'
    },
    async submit() {
      this.uploading = true
      this.errorMsg = ''

      const formData = new FormData()
      formData.append('video', this.selectedFile)
      formData.append('email', this.email.trim())

      try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData })
        const data = await res.json()

        if (!res.ok) {
          this.errorMsg = data.error || 'Upload failed.'
          return
        }

        this.$emit('job-started', data.job_id)
      } catch (err) {
        this.errorMsg = 'Network error. Is the server running?'
      } finally {
        this.uploading = false
      }
    },
  },
}
</script>

<style scoped>
.upload-form {
  width: 100%;
  max-width: 480px;
  padding: 36px 32px 32px;
}

/* Header */
.form-header {
  margin-bottom: 24px;
}

.form-title {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.3px;
}

/* Drop zone */
.drop-zone {
  border: 2px dashed #c5cdd5;
  border-radius: var(--radius-md);
  padding: 32px 20px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
  margin-bottom: 20px;
  background: transparent;
}

.drop-zone:hover,
.drop-zone--active {
  border-color: var(--accent);
  background: var(--accent-light);
}

.drop-zone--has-file {
  border-style: solid;
  border-color: var(--accent);
  padding: 16px 20px;
}

.drop-zone__icon {
  display: flex;
  justify-content: center;
  margin-bottom: 12px;
}

.drop-zone__text {
  color: var(--text-secondary);
  font-size: 16px;
  margin-bottom: 10px;
}

.drop-zone__rules {
  list-style: decimal;
  text-align: left;
  display: inline-block;
  color: var(--text-muted);
  font-size: 14px;
  line-height: 1.7;
  padding-left: 16px;
}

.drop-zone__preview {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.file-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--accent);
}

.file-name {
  color: var(--text-primary);
  font-weight: 500;
  font-size: 14px;
  max-width: 240px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-size {
  color: var(--text-muted);
  font-size: 12px;
}

.remove-btn {
  background: var(--error-light);
  border: none;
  border-radius: var(--radius-sm);
  color: var(--error);
  font-size: 13px;
  width: 28px;
  height: 28px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
}

.remove-btn:hover {
  background: rgba(220, 38, 38, 0.14);
}

.file-input-hidden {
  display: none;
}

/* Email group */
.email-group {
  margin-bottom: 20px;
}

.input-label {
  display: block;
  color: var(--text-label);
  font-size: 13px;
  font-weight: 500;
  margin-bottom: 6px;
}

/* Submit */
.btn-submit {
  width: 100%;
  padding: 13px;
  font-size: 16px;
  border-radius: var(--radius-md);
  letter-spacing: 0.2px;
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.4);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}

/* Error */
.error-msg {
  color: var(--error);
  font-size: 12px;
  text-align: center;
  margin-top: 14px;
  padding: 8px 12px;
  background: var(--error-light);
  border-radius: var(--radius-sm);
}
</style>
