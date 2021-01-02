<script>

import { Col, Container, Row, Button, Form, FormGroup, Input, Label } from 'sveltestrap';

	let out;
	const url = "http://127.0.0.1:8000/predict";

	function getPrediction(model_path = "./models/mobile_net_100_20201207-090439/mobile_net_100_20201207-090439", img_path = './dataset/test/PotatoHealthy1.JPG') {

		const payload = {};
		payload["model_path"] = model_path;
		payload["img_path"] = img_path;

		fetch(url, {
			method: "POST",
			headers: {
				"Accept": "application/json",
				"Content-Type": "application/json"
			},
			body: JSON.stringify(payload, ["message", "arguments", "type", "name"])
		})
		.then(d => d.json())
    	.then(d => {
			out = d
		});
		}

</script>

<main>
	<Container>
		<Form>
		<FormGroup>
    		<Label for="exampleFile">File</Label>
    		<Input type="file" name="file" id="exampleFile" oninput="predimg.src=window.URL.createObjectURL(this.files[0])" />
			<img id="predimg" src="" width="200px" height="200px" alt = " "/>
		</FormGroup>
	</Form>
		<Button on:click={getPrediction}>Get a prediction</Button>
	</Container>
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