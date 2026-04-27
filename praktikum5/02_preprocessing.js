// Paksa pakai CPU (atasi error WebGL)
tf.setBackend('cpu');

// Ambil dan bersihkan data
async function getData() {
  try {
    const response = await fetch(
      "https://raw.githubusercontent.com/pierpaolo28/Artificial-Intelligence-Projects/master/Google%20AI%20tools/tensorflow.js/swedish.json"
    );

    if (!response.ok) {
      throw new Error("Gagal mengambil data");
    }

    const insuranceData = await response.json();

    const cleaned = insuranceData
      .map(insurance => ({
        claims: insurance.X,
        payment: insurance.Y
      }))
      .filter(insurance =>
        insurance.claims != null && insurance.payment != null
      );

    return cleaned;

  } catch (error) {
    console.error("Error:", error);
    return [];
  }
}

// Jalankan saat halaman dibuka
document.addEventListener("DOMContentLoaded", async () => {
  const data = await getData();

  console.log("Data berhasil diambil:", data);

  // Ubah ke format grafik
  const values = data.map(d => ({
    x: d.claims,
    y: d.payment
  }));

  // Tampilkan scatterplot
  tfvis.render.scatterplot(
    { name: "Number of Claims vs Total Payment" },
    { values },
    {
      xLabel: "Claims",
      yLabel: "Payment",
      height: 300
    }
  );
});