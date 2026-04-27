function testModel(model, input, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  const [xs, preds] = tf.tidy(() => {
    const xs = input;

    // Prediksi (tanpa hardcode jumlah data)
    const preds = model.predict(xs.reshape([xs.shape[0], 1]));

    // Denormalisasi
    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  // Data hasil prediksi
  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  // Data asli
  const originalPoints = inputData.map(d => ({
    x: d.claims,
    y: d.payment
  }));

  // Visualisasi
  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"]
    },
    {
      xLabel: "Number of claims",
      yLabel: "Total Payment",
      height: 300
    }
  );
}