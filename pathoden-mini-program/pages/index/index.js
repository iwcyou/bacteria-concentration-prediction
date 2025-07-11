Page({
  data: {
    imagePath: '',
    resultText: '',
    loading: false,
  },

  chooseImage() {
    wx.chooseImage({
      count: 1,
      sizeType: ['original', 'compressed'],
      sourceType: ['album', 'camera'],
      success: (res) => {
        this.setData({
          imagePath: res.tempFilePaths[0],
          resultText: '',
        });
      }
    });
  },

  startDetection() {
    const { imagePath } = this.data;
    if (!imagePath) {
      wx.showToast({
        title: 'Please upload a photo first',
        icon: 'none'
      });
      return;
    }

    this.setData({ loading: true });

    wx.uploadFile({
      url: 'http://10.16.190.175:5000/predict',
      filePath: imagePath,
      name: 'file',
      success: (res) => {
        const data = JSON.parse(res.data);
        this.setData({
          resultText: data.suggestion,
          loading: false,
        });

        wx.pageScrollTo({
          selector: '#result',
          duration: 300
        });
      },
      fail: (err) => {
        console.error(err);
        wx.showToast({
          title: 'Detection failed',
          icon: 'none'
        });
        this.setData({ loading: false });
      }
    });
  }
})
