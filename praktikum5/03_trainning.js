function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({ inputShape: [1], units: 5, useBias: true }));
  model.add(tf.layers.dense({ units: 10, useBias: true }));
  model.add(tf.layers.dense({ units: 1, useBias: true }));

  return model;
}

async function trainModel(model, inputs, labels) {
  await tf.ready(); // pastikan backend siap

  model.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError',
    metrics: ['mse']
  });

  const batchSize = 32;
  const epochs = 50;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    validationSplit: 0.2,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Progress" },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    )
  });
}