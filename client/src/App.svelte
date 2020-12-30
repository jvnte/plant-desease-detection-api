<script>
	let out;
	const url = "http://127.0.0.1:8000/predict";

	function getPrediction() {
	fetch(url, {
		method: "POST",
		headers: {
			"Accept": "application/json",
			"Content-Type": "application/json"
		},
		body: JSON.stringify({model_path: './models/mobile_net_100_20201207-090439/mobile_net_100_20201207-090439', img_path: './dataset/test/PotatoHealthy1.JPG'})
	})
	.then(d => d.json())
    .then(d => {
		out = d
	});
	}
</script>

<main>
	<button on:click={getPrediction}>Get a prediction</button>
	{#if out}
	<h1>This is the outcome:</h1>
	<p>Target: {out.target}</p>
	<p>Prediction : {out.prediction}</p>
	{/if}
</main>

<style>
	main {
		text-align: center;
		padding: 1em;
		max-width: 240px;
		margin: 0 auto;
	}

	h1 {
		color: #ff3e00;
		text-transform: uppercase;
		font-size: 4em;
		font-weight: 100;
	}

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>