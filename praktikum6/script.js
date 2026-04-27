// script.js
let model;

async function setupModel() {
    // 1. Arsitektur Model
    model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    // 2. Kompilasi Model
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    // 3. Data Training (y = 2x - 1)
    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

    console.log("Sedang melatih model...");
    await model.fit(xs, ys, {epochs: 250}); 
    console.log("Model selesai dilatih!");
}

// Praktikum 5 & 6: Fungsi Prediksi
async function prediksi() {
    const nilaiInput = parseFloat(document.getElementById('inputData').value);
    if (isNaN(nilaiInput)) {
        alert("Masukkan angka dulu!");
        return;
    }
    const hasilOutput = model.predict(tf.tensor2d([nilaiInput], [1, 1]));
    const prediksiFinal = await hasilOutput.data();
    document.getElementById('hasil').innerText = `Hasil prediksi (y): ${prediksiFinal[0].toFixed(2)}`;
}

// WAJIB: Panggil fungsi setup agar model dibuat saat file ini dibaca
setupModel();