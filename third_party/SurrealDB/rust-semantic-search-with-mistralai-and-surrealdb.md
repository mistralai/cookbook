# Semantic search in Rust using SurrealDB and Mistral AI

This post demonstrates how to use the Rust SDK to store Mistral AI embeddings as [SurrealDB vectors](https:www.surrealdb.com/docs/surrealdb/reference-guide/vector-search), which can then be queried natively in SurrealQL to perform semantic search.

This guide uses Rust's [mistralai-client](https://crates.io/crates/mistralai-client) crate to generate embeddings, but the code below can be modified to suit [other languages](https://docs.mistral.ai/getting-started/clients/#rust) that have clients for Mistral's AI platform. If you are a Python user, check out [this page](https:www.surrealdb.com/docs/integrations/embeddings/mistral) in the documentation for another ready-made example.

## Setup

Setting up an embedded SurrealDB database requires no installation and can be done in just a few lines of code. After creating a new Cargo project with `cargo new project_name` and going into the project folder, add the following dependencies inside `Cargo.toml`:

```toml
anyhow = "1.0.98"
mistralai-client = "0.14.0"
serde = "1.0.219"
surrealdb = { version = "2.3", features = ["kv-mem"] }
tokio = "1.45.0"
```
<br />

You can add the same dependencies on the command line through a single command:

```
cargo add anyhow mistralai-client serde tokio surrealdb --features surrealdb/kv-mem
```
<br />

Setting up a SurrealDB database in Rust is as easy as calling the `connect` function with `"memory"` to instantiate an embedded database in memory. This code uses `anyhow` to allow the question mark operator to be used, but you can also just begin with `.unwrap()` everywhere and eventually move on to your own preferred error handling.

```rust
use anyhow::Error;
use surrealdb::engine::any::connect;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let db = connect("memory").await?;
    Ok(())
}
```
<br />

If you have a running Cloud or local instance, you can pass that path into the `connect()` function instead.

```rust
// Cloud address
let db = connect("wss://cloud-docs-068rp16e0hsnl62vgooa7omjks.aws-euw1.staging.surrealdb.cloud").await?;

// Local address
let db = connect("ws://localhost:8000").await?;
```
<br />

After connecting, we will select a namespace and database name, such as `ns` and `db`.

```rust
db.use_ns("ns").use_db("db").await?;
```
<br />

## Create a vector table and index

Next we'll create a table called `document` to store documents and embeddings, along with an index for the embeddings. The statements look like this:

```surql
DEFINE TABLE document;
DEFINE FIELD text ON document TYPE string;
DEFINE FIELD embedding ON document TYPE array<float>;
DEFINE INDEX hnsw_embed ON document FIELDS embedding HNSW DIMENSION 1024 DIST COSINE;
```
<br />

The important piece to understand is the relationship between the `embedding` field, a simple array of floats, and the index that we have given the name `hnsw_embed`. The size of the vector (1024 here) represents the number of dimensions in the embedding. This is to match Mistral AI's `mistral-embed` model, which uses [1024 as its length](https://docs.mistral.ai/getting-started/models/models_overview/#premier-models).

The [HNSW index](https:www.surrealdb.com/docs/surrealdb/reference-guide/vector-search#vector-indexes) is not strictly necessary to use the KNN operator (`<||>`) to find an embedding's closest neighbours, and for our small sample code we will use the simple [brute force method](https:www.surrealdb.com/docs/surrealql/operators#brute-force-method) which chooses [an algorithm](https:www.surrealdb.com/docs/surrealdb/reference-guide/vector-search#computation-on-vectors-vector-package-of-functions) such as Euclidean, Hamming, and so on. The following is the code that we will use, which uses the cosine of an embedding to find the four closest neighbours.

```surql
SELECT 
    text,
    vector::distance::knn() AS distance FROM document
    WHERE embedding <|4,COSINE|> $embeds
    ORDER BY distance;
```

As the dataset grows, however, the syntax can be changed to use [the HNSW index](https:www.surrealdb.com/docs/surrealql/operators#hnsw-method), by replacing an algorithm with a number that represents the size of the dynamic candidate list. This index is recommended when a small loss of accuracy is acceptable in order to preserve performance.

```surql
SELECT 
    text,
    vector::distance::knn() AS distance FROM document
    WHERE embedding <|4,40|> $embeds
    ORDER BY distance;
```

Another option is to use the [MTREE](https:www.surrealdb.com/docs/surrealql/operators#mtree-index-method) index method.

Inside the Rust SDK we can put all four of these inside a single `.query()` call and then add a line to see if there are errors inside any of them.

```rust
let mut res = db
    .query(
        "DEFINE TABLE document;
DEFINE FIELD text ON document TYPE string;
DEFINE FIELD embedding ON document TYPE array<float>;
DEFINE INDEX hnsw_embed ON document FIELDS embedding HNSW DIMENSION 1024 DIST COSINE;",
    )
    .await?;
for (index, error) in res.take_errors() {
    println!("Error in query {index}: {error}");
}
```
<br />

## Generate Mistral AI embeddings

At this point, you will need a [key](https://console.mistral.ai/api-keys) to interact with Mistral AI's platform. They offer a free tier for experimentation, after which you will be able to create a key to interact with it via the code below.

The code in this page will still work without a proper code, but the request to the Mistral AI API will end up returning the following error message.

```
Error: ApiError: 401 Unauthorized: {"detail":"Unauthorized"}
```
<br />

The best way to set the key is as an environment variable, which we will set to be a static called `KEY`. The client will look for one called `MISTRAL_API_KEY`, though you can change this when setting up the Mistral AI Rust client if you like.

```rust
// Looks for MISTRAL_API_KEY
let client = Client::new(Some(KEY.to_string()), None, None, None)?;
// Looks for OTHER_ENV_VAR
let client = Client::new(Some(KEY.to_string()), Some("OTHER_ENV_VAR".to_string()), None, None)?;
```

Using a `LazyLock` will let us call it via `std::env::var()` function the first time it is accessed. You can of course simply put it into a `const` for simplicity when first testing, but always remember to never hard-code API keys in your code in production.

```rust
static KEY: LazyLock<String> = LazyLock::new(|| {
    std::env::var("MISTRAL_API_KEY").unwrap()
});
```
<br />

And then run the code like this:

```bash
MISTRAL_API_KEY=whateverthekeyis cargo run
```
<br />

Or like this if you are using PowerShell on Windows.

```powershell
$env:MISTRAL_API_KEY = "whateverthekeyis"
cargo run
```
<br />

We can also create a `const MODEL` to hold the Mistral AI model used, which in this case is an `EmbedModel::MistralEmbed`.

```rust
const MODEL: EmbedModel = EmbedModel::MistralEmbed;
```

Inside `main()`, we will then [create a client](https://docs.rs/mistralai-client/0.14.0/mistralai_client/v1/client/struct.Client.html#method.new) from the `mistralai-client` crate.

```rust
let client = Client::new(Some(KEY.to_string()), None, None, None)?;
```
<br />


We'll use that to generate a Mistral AI embedding using the [`mistral-embed`](https://docs.mistral.ai/getting-started/models/models_overview/#premier-models) model. The `mistralai-client` has both sync and async functions that take a `Vec<String>`, and since SurrealDB uses the tokio runtime, we'll call the async `.embeddings_async()` method.

```rust
let input = vec!["Joram is the main character in the Darksword Trilogy.".to_string()];

let result = client.embeddings_async(MODEL, input, None).await?;
println!("{:?}", result);
```
<br />

The output in your console should show a massive amount of floats, 1024 of them to be precise. That's the embedding for this input!

## Store embeddings in database

Now that we have the embedding returned from the Mistral AI client, we can store it in the database. The [response](https://docs.rs/mistralai-client/0.14.0/mistralai_client/v1/embedding/struct.EmbeddingResponse.html) returned from the mistralai-client crate looks like this, with a `Vec` of `EmbeddingResponseDataItem` structs that hold a `Vec<f32>`.

```rust
pub struct EmbeddingResponse {
    pub id: String,
    pub object: String,
    pub model: EmbedModel,
    pub data: Vec<EmbeddingResponseDataItem>,
    pub usage: ResponseUsage,
}

pub struct EmbeddingResponseDataItem {
    pub index: u32,
    pub embedding: Vec<f32>,
    pub object: String,
}
```
<br />

We know that our simple request only returned a single embedding, so `.remove(0)` will do the job. In a more complex codebase you would probably opt for a match on `.get(0)` to handle any possible errors.

```rust
let embeds = result.data.remove(0).embedding;
```
<br />

There are a [number of ways](https:www.surrealdb.com/docs/sdk/rust/concepts/flexible-typing) to work with or avoid structs when using the Rust SDK, but we'll just go with two basic structs: one to represent the input into a `.create()` statement, which will implement `Serialize`, and another that implements `Deserialize` to show the result.

```rust
#[derive(Serialize)]
struct DocumentInput {
    text: String,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct Document {
    id: RecordId,
    embedding: Vec<f32>,
    text: String,
}
```
<br />

Once that is done, we can print out the created documents as a `Document` struct. We'll fiddle with the code a bit to have the `input` start as a `&str` which will be turned into a `String` in order to get the embedding, as well as to create a `Document` struct.

```rust
let input = "Octopuses solve puzzles and escape enclosures, showing advanced intelligence.";

let mut result = client
    .embeddings_async(MODEL, vec![input.to_string()], None)
    .await?;
let embeds = result.data.remove(0).embedding;
let in_db = db
    .create::<Option<Document>>("document")
    .content(DocumentInput {
        text: input.into(),
        embedding: embeds.to_vec(),
    })
    .await?;
println!("{in_db:?}");
```
<br />

We should now add some more `document` records. To do this, we'll move the logic to create them inside a function of its own. Since the `embeddings_async()` method takes a single `Vec<String>`, we'll first clone it to keep the original `Vec<String>` around, then zip it together with the embeddings returned so that they can be put into the database along with the original input.

```rust
async fn create_embeds(
    input: Vec<String>,
    db: &Surreal<Any>,
    client: &Client,
) -> Result<(), Error> {
    let cloned = input.clone();
    let embeds = client.embeddings_async(MODEL, input, None).await?;
    let zipped = cloned
        .into_iter()
        .zip(embeds.data.into_iter().map(|item| item.embedding));

    for (text, embeds) in zipped {
        let _in_db = db
            .create::<Option<Document>>("document")
            .content(DocumentInput {
                text,
                embedding: embeds,
            })
            .await?;
    }
    Ok(())
}
```
<br />

Then we'll create four facts for each of four topics: sea creatures, Korean and Japanese cities, historical figures, and planets of the Solar System (including the dwarf planet Ceres).

```rust
let embeds = [
    "Octopuses solve puzzles and escape enclosures, showing advanced intelligence.",
    "Sharks exhibit learning behavior, but their intelligence is instinct-driven.",
    "Sea cucumbers lack a brain and show minimal cognitive response.",
    "Clams have simple nervous systems with no known intelligent behavior.",
    //
    "Seoul is South Korea’s capital and a global tech hub.",
    "Sejong is South Korea’s planned administrative capital.",
    "Busan a major South Korean port located in the far southeast.",
    "Tokyo is Japan’s capital, known for innovation and dense population.",
    //
    "Wilhelm II was Germany’s last Kaiser before World War I.",
    "Cyrus the Great founded the Persian Empire with tolerant rule.",
    "Napoleon Bonaparte was a French emperor and brilliant military strategist.",
    "Aristotle was a Greek philosopher who shaped Western intellectual thought.",
    //
    "Venus’s atmosphere ranges from scorching surface to Earth-like upper clouds.",
    "Mars has a thin, cold atmosphere with seasonal dust storms.",
    "Ceres has a tenuous exosphere with sporadic water vapor traces.",
    "Saturn’s atmosphere spans cold outer layers to a deep metallic hydrogen interior",
]
.into_iter()
.map(|s| s.to_string())
.collect::<Vec<String>>();

create_embeds(embeds, &db, &client).await?;
```
<br />

## Semantic search

Finally let's perform semantic search over the embeddings in our database. We'll go with this query that uses the KNN operator to return the closest four matches to an embedding.

```surql
SELECT 
    text,
    vector::distance::knn() AS distance FROM document
    WHERE embedding <|4,COSINE|> $embeds
    ORDER BY distance;
```
<br />

You can customise this [with other algorithms](https:www.surrealdb.com/docs/surrealdb/reference-guide/vector-search#computation-on-vectors-vector-package-of-functions) such as Euclidean, Hamming, and so on.

We will then put this into a separate function called `ask_question()` which looks similar `create_embed()`, except that it first prints out its input, and then uses its embedding retrieved from Mistral to query the database against existing documents instead of creating a new document.

```rust
async fn ask_question(input: &str, db: &Surreal<Any>, client: &Client) -> Result<(), Error> {
    println!("{input}");
    let embeds = client
        .embeddings_async(MODEL, vec![input.to_string()], None)
        .await?
        .data
        .remove(0)
        .embedding;

    let mut response = db.query("SELECT text, vector::distance::knn() AS distance FROM document WHERE embedding <|4,COSINE|> $embeds ORDER BY distance;").bind(("embeds", embeds)).await?;
    let as_val: Value = response.take(0)?;
    println!("{as_val}\n");
    Ok(())
}
```
<br />

Finally, we will call this function a few times inside `main()` to confirm that the results are what we expect them to be, printing out the results of each so that we can eyeball them and make sure that they are what we expect them to be.

```rust
ask_question("Which Korean city is just across the sea from Japan?", &db, &client).await?;
ask_question("Who was Germany's last Kaiser?", &db, &client).await?;
ask_question("Which sea animal is most intelligent?", &db, &client).await?;
ask_question("Which planet's atmosphere has a part with the same temperature as Earth?", &db, &client).await?;
```
<br />

The output shows that the facts that fit most to our questions end up displayed first, with differing distance depending on how close the other facts were. Octopuses end up smarter than sharks (which is true), but the "learning behavior" part of our input does end up making sharks score pretty close. On the other extreme, Wilhelm II is clearly the only input that comes anywhere close to "Germany's last Kaiser", with Napoleon Bonaparte way behind. Poor Aristotle doesn't make it into any results, with "Sejong is South Korea's planned administrative capital" slightly closer semantically in terms of "Who was Germany's last Kaiser".

```
Which Korean city is just across the sea from Japan?
[{ distance: 0.19170371029549582f, text: 'Busan is a major South Korean port located in the far southeast.' }, { distance: 0.2399314515762122f, text: 'Tokyo is Japan’s capital, known for innovation and dense population.' }, { distance: 0.2443623703771407f, text: 'Sejong is South Korea’s planned administrative capital.' }, { distance: 0.24488082839731895f, text: 'Seoul is South Korea’s capital and a global tech hub.' }]

Who was Germany's last Kaiser?
[{ distance: 0.11228576780228805f, text: 'Wilhelm II was Germany’s last Kaiser before World War I.' }, { distance: 0.2957177300085634f, text: 'Napoleon Bonaparte was a French emperor and brilliant military strategist.' }, { distance: 0.34394473621670896f, text: 'Cyrus the Great founded the Persian Empire with tolerant rule.' }, { distance: 0.34911517400935843f, text: 'Sejong is South Korea’s planned administrative capital.' }]

Which sea animal is most intelligent?
[{ distance: 0.2342596053829904f, text: 'Octopuses solve puzzles and escape enclosures, showing advanced intelligence.' }, { distance: 0.24131327939924785f, text: 'Sharks exhibit learning behavior, but their intelligence is instinct-driven.' }, { distance: 0.2426242772516931f, text: 'Clams have simple nervous systems with no known intelligent behavior.' }, { distance: 0.24474598154128135f, text: 'Sea cucumbers lack a brain and show minimal cognitive response.' }]

Which planet's atmosphere has a part with the same temperature as Earth?
[{ distance: 0.20653440713083582f, text: 'Venus’s atmosphere ranges from scorching surface to Earth-like upper clouds.' }, { distance: 0.23354208810464594f, text: 'Mars has a thin, cold atmosphere with seasonal dust storms.' }, { distance: 0.24560810032473468f, text: 'Saturn’s atmosphere spans cold outer layers to a deep metallic hydrogen interior' }, { distance: 0.2761595357544341f, text: 'Ceres has a tenuous exosphere with sporadic water vapor traces.' }]
```
<br />

As the database grows, you could also change the `<|4,COSINE|>` part of the query to something like `<|4,40|>` to see the results using the HNSW index instead of the brute force method.

Finally, here is all of the code for you to run and modify as you wish.

```rust
use std::sync::LazyLock;

use anyhow::Error;
use mistralai_client::v1::{client::Client, constants::EmbedModel};
use serde::{Deserialize, Serialize};
use surrealdb::{
    RecordId, Surreal, Value,
    engine::any::{Any, connect},
};

static KEY: LazyLock<String> = LazyLock::new(|| std::env::var("MISTRAL_API_KEY").unwrap());

// Experiment plan
const MODEL: EmbedModel = EmbedModel::MistralEmbed;

#[derive(Serialize)]
struct DocumentInput {
    text: String,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct Document {
    id: RecordId,
    embedding: Vec<f32>,
    text: String,
}

async fn create_embeds(
    input: Vec<String>,
    db: &Surreal<Any>,
    client: &Client,
) -> Result<(), Error> {
    let cloned = input.clone();
    let embeds = client.embeddings_async(MODEL, input, None).await?;
    let zipped = cloned
        .into_iter()
        .zip(embeds.data.into_iter().map(|item| item.embedding));

    for (text, embeds) in zipped {
        let _in_db = db
            .create::<Option<Document>>("document")
            .content(DocumentInput {
                text,
                embedding: embeds,
            })
            .await?;
    }
    Ok(())
}

async fn ask_question(input: &str, db: &Surreal<Any>, client: &Client) -> Result<(), Error> {
    println!("{input}");
    let embeds = client
        .embeddings_async(MODEL, vec![input.to_string()], None)
        .await?
        .data
        .remove(0)
        .embedding;

    let mut response = db.query("SELECT text, vector::distance::knn() AS distance FROM document WHERE embedding <|4,COSINE|> $embeds ORDER BY distance;").bind(("embeds", embeds)).await?;
    let as_val: Value = response.take(0)?;
    println!("{as_val}\n");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let db = connect("memory").await.unwrap();

    db.use_ns("ns").use_db("db").await.unwrap();

    let mut res = db
        .query(
            "DEFINE TABLE document;
             DEFINE FIELD text ON document TYPE string;
             DEFINE FIELD embedding ON document TYPE array<float>;
             DEFINE INDEX hnsw_embed ON document FIELDS embedding HNSW DIMENSION 1024 DIST COSINE;",
        )
        .await
        .unwrap();
    for (index, error) in res.take_errors() {
        println!("Error in query {index}: {error}");
    }

    let client = Client::new(Some(KEY.to_string()), None, None, None)?;

    let embeds = [
        "Octopuses solve puzzles and escape enclosures, showing advanced intelligence.",
        "Sharks exhibit learning behavior, but their intelligence is instinct-driven.",
        "Sea cucumbers lack a brain and show minimal cognitive response.",
        "Clams have simple nervous systems with no known intelligent behavior.",
        //
        "Seoul is South Korea’s capital and a global tech hub.",
        "Sejong is South Korea’s planned administrative capital.",
        "Busan is a major South Korean port located in the far southeast.",
        "Tokyo is Japan’s capital, known for innovation and dense population.",
        //
        "Wilhelm II was Germany’s last Kaiser before World War I.",
        "Cyrus the Great founded the Persian Empire with tolerant rule.",
        "Napoleon Bonaparte was a French emperor and brilliant military strategist.",
        "Aristotle was a Greek philosopher who shaped Western intellectual thought.",
        //
        "Venus’s atmosphere ranges from scorching surface to Earth-like upper clouds.",
        "Mars has a thin, cold atmosphere with seasonal dust storms.",
        "Ceres has a tenuous exosphere with sporadic water vapor traces.",
        "Saturn’s atmosphere spans cold outer layers to a deep metallic hydrogen interior",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect::<Vec<String>>();

    create_embeds(embeds, &db, &client).await?;

    ask_question("Which Korean city is just across the sea from Japan?", &db, &client).await?;
    ask_question("Who was Germany's last Kaiser?", &db, &client).await?;
    ask_question("Which sea animal is most intelligent?", &db, &client).await?;
    ask_question("Which planet's atmosphere has a part with the same temperature as Earth?", &db, &client).await?;

    Ok(())
}
```

<br />