async function run() {
  try {
    await tf.ready(); // pastikan TensorFlow siap

    // Ambil data
    const data = await getData();

    if (!data || data.length === 0) {
      console.error("Data kosong");
      return;
    }

    // Visualisasi data awal
    const values = data.map(d => ({
      x: d.claims,
      y: d.payment
    }));

    tfvis.render.scatterplot(
      { name: "Number of claims vs Total Payment" },
      { values },
      {
        xLabel: "Number of claims",
        yLabel: "Total Payment",
        height: 300
      }
    );

    // Buat model
    const model = createModel();
    tfvis.show.modelSummary({ name: "Model Summary" }, model);

    // Preprocessing
    const tensorData = convertToTensor(data);
    const { inputs, labels } = tensorData;

    // Training
    await trainModel(model, inputs, labels);
    console.log("Done Training");

    // Testing
    testModel(model, inputs, data, tensorData);

  } catch (error) {
    console.error("Error di run():", error);
  }
}

// Jalankan saat halaman load
document.addEventListener("DOMContentLoaded", () => {
  run();
});