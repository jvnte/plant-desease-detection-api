<script>

import { Button } from 'sveltestrap';

	let out, avatar, avatar_name, fileinput;
	const url = "http://127.0.0.1:8000/predict";

	function getPrediction(model_path, img_path) {

		fetch(url, {
			method: "POST",
			headers: {
				"Content-Type": "application/json"
			},
			body: JSON.stringify({model_path: model_path, img_path: img_path})
		})
		.then(d => d.json())
    	.then(d => {
			out = d
		});
		}

	const onFileSelected =(e)=>{
		let image = e.target.files[0];
			avatar_name = image.name;
            let reader = new FileReader();
            reader.readAsDataURL(image);
            reader.onload = e => {
                 avatar = e.target.result
            };
	}

</script>

<div id="app">
	<h1>Upload Image</h1>
		{#if avatar}
        <img class="avatar" src="{avatar}" alt="d" />
		<p>{avatar_name}</p>
		{:else}
		<img class="avatar" src="https://cdn4.iconfinder.com/data/icons/small-n-flat/24/user-alt-512.png" alt="" />
        {/if}
		<img class="upload" src="https://static.thenounproject.com/png/625182-200.png" alt="" on:click={()=>{fileinput.click();}} />
		<div class="chan" on:click={()=>{fileinput.click();}}>Choose Image</div>
		<input style="display:none" type="file" accept=".jpg" on:change={(e)=>onFileSelected(e)} bind:this={fileinput} >
		{#if avatar_name}
		<Button on:click={() => getPrediction('./models/vgg16_100_20201206-195647', './dataset/test/' + avatar_name)}>Get a prediction</Button>
		{/if}
	{#if out}
	<h1>This is the outcome:</h1>
		<p>Prediction : {out.prediction}</p>
	{/if}

</div>

<style>

	#app{
	display:flex;
		align-items:center;
		justify-content:center;
		flex-flow:column;
	}

	h1{
		padding-top: 50px;
	}

	.upload{
		display:flex;
	    height:50px;
		width:50px;
		cursor:pointer;
	}
	.avatar{
		display:flex;
		height:200px;
		width:200px;
	}

</style>