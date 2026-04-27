function convertToTensor(data) {
  return tf.tidy(() => {

    if (!data || data.length === 0) {
      throw new Error("Data kosong");
    }

    // 1. Shuffle data
    tf.util.shuffle(data);

    // 2. Pisahkan input & label
    const inputs = data.map(d => d.claims);
    const labels = data.map(d => d.payment);

    // 3. Ubah ke tensor
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // 4. Normalisasi
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin).add(1e-7));

    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin).add(1e-7));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin
    };
  });
}