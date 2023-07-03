# Prepoznavanje pasmine pasa

**Autori:** Toni Baskijera i Martina Sirotić

**Mentor:** doc. dr. sc. Marina Ivašić Kos

**Kolegij:** Strojno i dubinsko učenje

**Akademska godina:** 2022/2023.

## Tablica sadržaja

- [Uvod](#uvod)
- [Korišteni podaci i razvojno okruženje](#korišteni-podaci-i-razvojno-okruženje)
- [Opis korištenih biblioteka i metoda](#opis-korištenih-biblioteka-i-metoda)
- [Izvedba eksperimenta](#izvedba-eksperimenta)
- [Rezultati i primjeri predikcija](#rezultati-i-primjeri-predikcija)
- [Literatura](#literatura)

## Uvod

### Zadatak

Zadatak ovog projekta je razvijanje modela strojnog učenja koji identificira pasminu psa sa slike. Dakle, obučavanjem na opsežnom skupu podataka, model će uz pomoć MobileNet arhitekture dubinskog učenja steći sposobnost prepoznavanja različitih pasmina pasa i primijeniti to znanje na nove slike koje mu korisnik proslijedi. Detaljno će se objasniti cjelokupni proces strojnog učenja, uključujući korištene podatke, primjenjene metode i rezultate. Na kraju će se provesti evaluacija, analiza i usporedba spomenutih rezultata.

### Motivacija i pregled područja

Klasifikacija pasmina pasa oduvijek je bila intrigantno područje proučavanja za ljubitelje pasa, uzgajivače, veterinare i istraživače. Sposobnost točne identifikacije i razlikovanja između različitih pasmina pasa igra ključnu ulogu u više područja.

Doprinosi području dobrobiti životinja pomažući u razvoju ciljane zdravstvene skrbi i planova liječenja. Različite pasmine imaju različitu osjetljivost na određene bolesti i medicinska stanja. Stoga sposobnost točne identifikacije pasmine psa može pomoći veterinarima u pružanju odgovarajuće skrbi i smanjenju rizika od pogrešne dijagnoze ili netočnog liječenja.

Nadalje, identifikacija pasmina pasa ključna je za uzgajivače i entuzijaste koji su uključeni u programe uzgoja. Točna klasifikacija pomaže u održavanju integriteta i čistoće pasmina, osiguravajući odgovorne i informirane prakse uzgoja. Također pomaže u prepoznavanju prikladnih parova za parenje, povećavajući šanse za stvaranje zdravog potomstva sa željenim osobinama.

Osim toga, potražnja za udomljavanjem kućnih ljubimaca raste u cijelom svijetu. Mnogi pojedinci i obitelji traže posebne pasmine pasa koje odgovaraju njihovom načinu života, životnim uvjetima i preferencijama. AI model sposoban za točno prepoznavanje i klasificiranje pasmina pasa može pojednostaviti proces udomljavanja, olakšavajući spajanje potencijalnih vlasnika s njihovim željenim pasminama učinkovitije i djelotvornije.

S nedavnim napretkom u umjetnoj inteligenciji i strojnom učenju, postoji prilika za automatizaciju procesa kroz razvoj modela strojnog učenja koji može točno klasificirati pasmine pasa.

## Korišteni podaci i razvojno okruženje

Skup podataka koji će se koristiti za razvijanje modela, je Standford Dogs Dataset [[1]](<https://www.tensorflow.org/datasets/catalog/stanford_dogs>). Sadrži slike 120 pasmina pasa iz cijelog svijeta. Ovaj skup podataka izgrađen je korištenjem slika i komentara s ImageNeta u svrhu kategorizacije slika. Izvorno je prikupljen za finu kategorizaciju slike, što je izazovan problem jer određene pasmine pasa imaju gotovo identične značajke ili se razlikuju u boji i dobi.

Opis dataseta:

- Domena - računalni vid i strojno učenje
- Tip - slikovni podaci
- Broj primjera za učenje (slika) - 20.580
- Dimenzija - svaka slika je različitih dimenzija
- Anotacije - oznake klasa, granični okviri

Za razvoj modela i cjelokupni proces koristiti će se Google Collab Pro, koji je baziran na Jupyter bilježnici i predstavlja moćan i praktičan alat za izvođenje Python koda u oblaku, posebno kada su u pitanju zadaci za analizu podataka, strojno učenje i duboko učenje.

## Opis korištenih biblioteka i metoda

Korištene biblioteke, moduli i metode korištene u projektu, uključujući one namijenjene dubokom učenju, radu s podacima, datotekama i operativnim sustavom i manipulacijom slikama, možemo isčitati na početku Jupyter bilježnice gdje se one i uvoze:

```python
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from PIL import Image

from keras.applications import MobileNetV2
from keras.models import load_model, Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

from tensorboard import notebook

from sklearn.metrics import classification_report
```

Pojašnjenje uloga biblioteka, modula i metoda:

- `tensorflow`: Razvojni okvir za strojno učenje
- `tensorflow_datasets`: Predefinirani skupovi podataka za strojno učenje.
- `numpy`: Matematičke operacije nad višedimenzionalnim nizovima i matricama
- `matplotlib.pyplot`: Alat za crtanje i vizualizaciju u Pythonu
- `datetime`: Manipulacija datumima i vremenima
- `os`: Interakcija s operacijskim sustavom, navigacija po direktorijima i rad s datotekama
- `PIL.Image`: Otvaranje, manipulacija i spremanje slika različitih formata
- `keras.applications.MobileNetV2`: Prethodno naučeni model MobileNetV2
- `keras.models.load_model`: Funkcija za učitavanje prethodno naučenih modela
- `keras.models.Sequential`: Klasa za definiranje sekvencijalnih modela
- `keras.layers.Flatten`: Ravnanje (flatten) sloj
- `keras.layers.Dense`: Gusti (dense) sloj
- `keras.layers.Conv2D`: Konvolucijski 2D sloj
- `keras.layers.MaxPool2D`: Maksimalno grupiranje (pooling) 2D sloj
- `keras.layers.Dropout`: Sloj ispuštanja (dropout)
- `keras.layers.GlobalAveragePooling2D`: Globalno prosječno grupiranje (pooling) 2D sloj
- `keras.optimizers.Adam`: Adam optimizator
- `keras.callbacks.ModelCheckpoint`: Callback za spremanje najboljeg modela tijekom treniranja
- `keras.callbacks.TensorBoard`: Callback za vizualizaciju napretka treniranja pomoću TensorBoard-a
- `tensorboard.notebook`: Alati za rad s TensorBoard-om u Jupyter bilježnici
- `google.colab.drive`: Funkcionalnost za montiranje i pristupanje Google Drive-u u Google Colab okruženju
- `sklearn.metrics.classification_report`:Služi za generiranje klasifikacijskog izvještaja

Arhitektura modela koja će se koristiti za model, kao što je već i spomenuto, naziva se MobileNet. MobileNet je arhitektura konvolucijske neuronske mreže posebno dizajnirana za mobilne uređaje i ugrađene sustave s ograničenim računalnim resursima. Razvijen je kako bi riješio izazov implementacije dubokih modela učenja na uređajima s malom memorijom i procesorskom snagom, uz održavanje visoke točnosti. MobileNet postiže ovo korištenjem dubinski odvojenih konvolucija, koje razdvajaju standardnu konvolucijsku operaciju na zasebne dubinske i točkaste konvolucije. Ovaj pristup značajno smanjuje broj parametara i računalnih operacija potrebnih za obradu, rezultirajući kompaktnim i učinkovitim modelom.

Razlog iz kojeg smo odabrali baš MobileNet arhitekturu je taj što bi kasnije model koji će se istrenirati htjeli upotrijebiti, odnosno koristiti kao glavnu funkcionalnost unutar mobilne aplikacije, koje su inače područje u kojem imamo nešto više iskustva.

## Izvedba eksperimenta

Za početak, potrebno je montirati Google Drive na direktorij `/content/drive`, kojeg ćemo koristiti za spremanje podataka:

```python
drive.mount('/content/drive')
```

U sljedećem dijelu koda definirane su funkcije za učitavanje i obradu skupa podataka:

```python
def load_dataset():
    (ds_train, ds_test), ds_info = tfds.load('stanford_dogs',
                                             split=['train', 'test'],
                                             shuffle_files=True,
                                             as_supervised=False,
                                             with_info=True,
                                             data_dir='/content/drive/MyDrive/dataset/tfds')
    return ds_train, ds_test, ds_info

def calculate_total_images(ds_train, ds_test):
    total_train_images = len(ds_train)
    total_test_images = len(ds_test)
    total_images = total_train_images + total_test_images
    return total_images

def preprocess(data, image_size, num_labels, cast=True, resize=True, normalize=True, one_hot=True):
    processed_image = data['image']
    label = data['label']
    if cast:
        processed_image = tf.cast(processed_image, tf.float32)
    if resize:
        processed_image = tf.image.resize(processed_image, image_size, method='nearest')
    if normalize:
        processed_image = processed_image / 255.
    if one_hot:
        label = tf.one_hot(label, num_labels)
    return processed_image, label

def prepare(dataset, image_shape, num_classes, batch_size=None):
    dataset = dataset.map(lambda x: preprocess(x, image_shape[0:-1], num_classes), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

```

`load_dataset()`: Ova funkcija učitava skup podataka "stanford_dogs" iz TensorFlow Datasets (tfds). Dohvaća podatke za trening i testiranje, informacije o skupu podataka i postavlja direktorij podataka na '/content/drive/MyDrive/dataset/tfds'. Vraća učitane skupove podataka za trening, testiranje i informacije o skupu podataka.

`calculate_total_images(ds_train, ds_test)`: Ova funkcija izračunava ukupan broj slika u skupu podataka. Prima skupove podataka za trening i testiranje i vraća ukupan broj slika uključujući slike za trening i testiranje.

`preprocess(data, image_size, num_labels, cast=True, resize=True, normalize=True, one_hot=True)`: Ova funkcija provodi predprocesiranje podataka. Prima podatke (slike i oznake), veličinu slike, broj oznaka, te opcionalne argumente za pretvorbu tipa podataka (cast), promjenu veličine slike (resize), normalizaciju slike (normalize) i pretvorbu oznaka u one-hot reprezentaciju. Izvodi odgovarajuće predobradne korake na slikama i oznakama i vraća predobrađene slike i oznake.

`prepare(dataset, image_shape, num_classes, batch_size=None)`: Ova funkcija priprema skup podataka za treniranje ili testiranje. Prima skup podataka, oblik slike, broj klasa i opcionalni argument za veličinu grupa (batch_size). Koristi spomenuta funkciju `preprocess()` za predobradu svakog podatka u skupu podataka, pohranjuje predobrađene podatke u predmemoriju (cache), grupira ih u grupe određene veličine (batch_size) ako je to specificirano, i povećava učinkovitost pri dohvatu podataka koristeći `prefetch()` metodu. Vraća pripremljeni skup podataka.

Tada skup podataka konačno možemo i preuzeti koristeći funkciju koju smo definirali:

```
ds_ train, ds_test, ds_info = load_dataset()
```

Ispis svih 120 pasmina:

```
load_labels()

['chihuahua',
 'japanese_spaniel',
 'maltese_dog',
 'pekinese',
 'shih-tzu',
 'blenheim_spaniel',
 'papillon',
 'toy_terrier',
 'rhodesian_ridgeback',
 'afghan_hound',
 'basset',
 'beagle',
 'bloodhound',
 'bluetick',
 'black-and-tan_coonhound',
 'walker_hound',
 'english_foxhound',
 'redbone',
 'borzoi',
 'irish_wolfhound',
 'italian_greyhound',
 'whippet',
 'ibizan_hound',
 'norwegian_elkhound',
 'otterhound',
 'saluki',
 'scottish_deerhound',
 'weimaraner',
 'staffordshire_bullterrier',
 'american_staffordshire_terrier',
 'bedlington_terrier',
 'border_terrier',
 'kerry_blue_terrier',
 'irish_terrier',
 'norfolk_terrier',
 'norwich_terrier',
 'yorkshire_terrier',
 'wire-haired_fox_terrier',
 'lakeland_terrier',
 'sealyham_terrier',
 'airedale',
 'cairn',
 'australian_terrier',
 'dandie_dinmont',
 'boston_bull',
 'miniature_schnauzer',
 'giant_schnauzer',
 'standard_schnauzer',
 'scotch_terrier',
 'tibetan_terrier',
 'silky_terrier',
 'soft-coated_wheaten_terrier',
 'west_highland_white_terrier',
 'lhasa',
 'flat-coated_retriever',
 'curly-coated_retriever',
 'golden_retriever',
 'labrador_retriever',
 'chesapeake_bay_retriever',
 'german_short-haired_pointer',
 'vizsla',
 'english_setter',
 'irish_setter',
 'gordon_setter',
 'brittany_spaniel',
 'clumber',
 'english_springer',
 'welsh_springer_spaniel',
 'cocker_spaniel',
 'sussex_spaniel',
 'irish_water_spaniel',
 'kuvasz',
 'schipperke',
 'groenendael',
 'malinois',
 'briard',
 'kelpie',
 'komondor',
 'old_english_sheepdog',
 'shetland_sheepdog',
 'collie',
 'border_collie',
 'bouvier_des_flandres',
 'rottweiler',
 'german_shepherd',
 'doberman',
 'miniature_pinscher',
 'greater_swiss_mountain_dog',
 'bernese_mountain_dog',
 'appenzeller',
 'entlebucher',
 'boxer',
 'bull_mastiff',
 'tibetan_mastiff',
 'french_bulldog',
 'great_dane',
 'saint_bernard',
 'eskimo_dog',
 'malamute',
 'siberian_husky',
 'affenpinscher',
 'basenji',
 'pug',
 'leonberg',
 'newfoundland',
 'great_pyrenees',
 'samoyed',
 'pomeranian',
 'chow',
 'keeshond',
 'brabancon_griffon',
 'pembroke',
 'cardigan',
 'toy_poodle',
 'miniature_poodle',
 'standard_poodle',
 'mexican_hairless',
 'dingo',
 'dhole',
 'african_hunting_dog']
```

Prikaz slika:

```
tfds.show_examples(ds_train, ds_info)
```

![Examples](https://raw.githubusercontent.com/tbaskijera/dog-breed-detection/main/images/examples.jpg)

Zatim ćemo definirati nekoliko konstanti koje ćemo koristiti:

```python
IMG_SHAPE = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_SHAPE, IMG_SHAPE, IMG_CHANNELS)
BATCH_SIZE = 32
NUM_BREEDS = len(load_labels())
```

- `IMG_SHAPE` definira oblik slika koje će se koristiti u modelu. U ovom slučaju, postavljen je na 224, što znači da će sve slike biti skalirane na dimenzije 224x224 piksela
- `IMG_CHANNELS` definira broj kanala slike. Ovdje je postavljen na 3, što označava tri kanala: crveni, zeleni i plavi (RGB)
- `INPUT_SHAPE` definira oblik ulaznih podataka za model. Ova varijabla je tuple koja sadrži dimenzije slike (IMG_SHAPE), broj kanala (IMG_CHANNELS) i stoga ima oblik (224, 224, 3)
- `BATCH_SIZE` predstavlja veličinu grupa (batch size) koja će se koristiti tijekom obuke modela. Ovdje je postavljen na 32, što znači da će se 32 primjera koristiti za svaku iteraciju tijekom obuke
- `NUM_BREEDS` predstavlja broj različitih pasmina pasa u skupu podataka. Ova varijabla dobiva vrijednost pomoću funkcije `len(load_labels())`, koja učitava oznake pasmina pasa i računa njihov broj

U nastavku se stvaraju skupovi za treniranje i testiranje:

```python
train_set = prepare(ds_train, INPUT_SHAPE, NUM_BREEDS, batch_size=BATCH_SIZE)
test_set = prepare(ds_test, INPUT_SHAPE, NUM_BREEDS, batch_size=BATCH_SIZE)
```

- `train_set` predstavlja skup za treniranje modela. Ova varijabla dobiva vrijednost pozivom funkcije `prepare` s parametrima `ds_train` (skup podataka za treniranje), `INPUT_SHAPE` (ulazni oblik podataka), `NUM_BREEDS` (broj pasmina) i `batch_size` (veličina grupe).
- `test_set` predstavlja skup za testiranje modela. Ova varijabla dobiva vrijednost pozivom funkcije `prepare` s parametrima `ds_test` (skup podataka za testiranje), `INPUT_SHAPE` (ulazni oblik podataka), `NUM_BREEDS` (broj pasmina) i `batch_size` (veličina grupe).

Model koji ćemo koristiti definirati ćemo isto tako u zasebnoj funkciji:

```python
def mobileNetV2(image_shape, num_classes, lr=0.001):
    base_model = MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

- Funkcija `mobileNetV2` prima tri parametra kao ulaz: `image_shape` (oblik slike), `num_classes` (broj klasa) i opcionalni parametar `lr` (stopa učenja) s zadanom vrijednošću `0.001`.
- Funkcija stvara temeljni model koristeći MobileNetV2 arhitekturu pozivajući funkciju `MobileNetV2` iz Keras biblioteke. Taj je model unaprijed istreniran na ImageNet skupu podataka i koristit će se kao ekstraktor značajki.
- Parametar `input_shape` MobileNetV2 modela postavljen je na vrijednost `image_shape` koja je proslijeđena funkciji `mobileNetV2`.
- Parametar `include_top` postavljen je na `False` kako bi se isključili potpuno povezani slojevi na vrhu MobileNetV2 modela, što nam omogućuje da kasnije dodamo svoje prilagođene slojeve.
- Parametar `weights` postavljen je na `'imagenet'`, što učitava unaprijed obučene težine MobileNetV2 modela obučenog na ImageNet skupu podataka.
- Svojstvo `trainable` temeljnog modela postavljeno je na `False`, što zamrzava težine temeljnog modela tijekom obuke, sprječavajući njihovo ažuriranje.
- Stvara se novi sekvencijalni model spajajući temeljni model i dodatne slojeve. Temeljni model služi kao početni dio modela i izvlači relevantne značajke iz ulaznih slika.
- Nakon temeljnog modela dodaje se sloj `GlobalAveragePooling2D`. Taj sloj uzima izlaz temeljnog modela i izračunava prosjek svake značajne mape, smanjujući prostorne dimenzije značajki.
- Izlaz sloja `GlobalAveragePooling2D` zatim se prosljeđuje sloju `Dense`. Taj sloj je potpuno povezan sloj s `num_classes` jedinica, što odgovara broju klasa u zadatku klasifikacije.
- Aktivacijska funkcija koja se koristi za sloj `Dense` je `'softmax'`, što proizvodi vjerojatnosnu distribuciju nad klasama, koja ukazuje na vjerojatnost svake od klasa
- Model se kompajlira pomoću metode `compile`. Optimizator je postavljen na Adam s stopom učenja `lr` koja je zadana kao parametar. Funkcija gubitka postavljena je na `'categorical_crossentropy'`, što se često koristi za višeklasnu klasifikaciju.
- Metrike za evaluaciju tijekom obuke postavljene su na `'accuracy'`, što mjeri točnost predikcija modela u usporedbi s istinskim oznakama.
- Na kraju, funkcija vraća kompilirani model.

U nastavku ćemo definirati još nekoliko funkcija:

```python
def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = f.read().splitlines()
    return labels

def create_model(input_shape, num_classes, learning_rate):
    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate(model, test_dataset):
    loss, accuracy = model.evaluate(test_dataset)
    print('Loss:', loss)
    print('Accuracy:', accuracy)

def generate_classification_report(model, dataset, num_classes, target_names):
    true_labels = []
    predicted_labels = []

    for image, label in dataset:
        true_labels.extend(label.numpy())
        prediction = model.predict(image)
        predicted_label = np.argmax(prediction, axis=1)[0]
        predicted_labels.append(predicted_label)

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    report = classification_report(true_labels, predicted_labels, target_names=target_names)
    return report

def training(train_dataset, test_dataset, model_dir, checkpoint_dir, log_dir, input_shape, num_classes, learning_rate, batch_size, epochs):
    model = create_model(input_shape, num_classes, learning_rate)

    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_dir, save_weights_only=True, save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)

    train_dataset = prepare(train_dataset, input_shape, num_classes, batch_size)
    test_dataset = prepare(test_dataset, input_shape, num_classes, batch_size)

    history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs,
                        callbacks=[checkpoint_callback, tensorboard_callback])

    model.save(model_dir)

    return history

def predict(model, image, labels, top_k=3):
    processed_image = preprocess(image, input_shape[0:-1], num_classes, resize=True, normalize=True, one_hot=False)
    processed_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(processed_image)[0]
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_predictions = predictions[top_indices]
    top_labels = [labels[i] for i in top_indices]
    return top_labels, top_predictions

def load_image(image_url, target_size):
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize(target_size)
    return image

def make_predictions(image_url='https://httpstatusdogs.com/404-not-found'):
    image = load_image(image_url, input_shape[0:-1])
    top_labels, top_predictions = predict(model, image, labels, top_k=3)

    plt.imshow(image)
    plt.axis('off')
    plt.show()

    for label, prediction in zip(top_labels, top_predictions):
        print(f'{label}: {prediction}')

```

- `load_labels(label_file)`: Ova funkcija prima putanju do datoteke s oznakama (`label_file`) kao ulaz. Čita sadržaj datoteke i razdvaja ga u listu oznaka. Zatim vraća listu oznaka.

- `create_model(input_shape, num_classes, learning_rate)`: Ova funkcija stvara model za klasifikaciju slika. Koristi MobileNetV2 model kao temeljni model s određenom ulaznom veličinom (`input_shape`). Na temelju tog modela dodaje globalni sloj sažimanja (`GlobalAveragePooling2D`), gusto povezani sloj s 256 jedinica i aktivacijskom funkcijom ReLU, sloj isključivanja (`Dropout`) s stopom isključivanja od 0.5 i gusti izlazni sloj s `num_classes` jedinica i aktivacijskom funkcijom softmax. Funkcija kompajlira model s Adam optimizatorom koristeći određenu stopu učenja, kategoričku križnu entropiju kao funkciju gubitka (loss) i točnost (accuracy) kao metriku. Zatim vraća kompilirani model.

- `evaluate(model, test_dataset)`: Ova funkcija ocjenjuje performanse naučenog modela na testnom skupu podataka. Računa gubitak (loss) i točnost (accuracy) modela na testnom skupu podataka koristeći metodu evaluate modela. Funkcija ispisuje izračunati gubitak i točnost.

- `generate_classification_report(model, dataset, num_classes, target_names)`: Ova funkcija generira izvještaj o klasifikaciji na temelju predikcija zadanim modelom na skupu podataka. Prima naučeni `model`, `dataset` na kojem će se model evaluirati, `num_classes` (broj klasa) i `target_names` (lista imena za svaku ciljnu klasu) kao parametre.

- `training(train_dataset, test_dataset, model_dir, checkpoint_dir, log_dir, input_shape, num_classes, learning_rate, batch_size, epochs)`: Ova funkcija izvodi treniranje modela. Prima skup podataka za treniranje (`train_dataset`) i skup podataka za testiranje (`test_dataset`), zajedno s različitim konfiguracijskim parametrima za treniranje kao što su `model_dir` (direktorij za spremanje naučenog modela), `checkpoint_dir` (direktorij za spremanje kontrolnih točaka modela), `log_dir` (direktorij za TensorBoard zapise), `input_shape` (ulazni oblik), `num_classes` (broj klasa), `learning_rate` (stopa učenja), `batch_size` (veličina grupe) i `epochs` (broj epoha). Prvo stvara model koristeći funkciju `create_model`. Zatim postavlja povratne pozive (callbacks) za spremanje kontrolnih točaka modela i zapisivanje zapisa u TensorBoard. Nakon toga priprema skupove podataka za treniranje i testiranje koristeći funkciju `prepare`. Na kraju trenira model koristeći metodu `fit`, prosljeđujući skup podataka za treniranje kao trening podatke, skup podataka za testiranje kao validacijske podatke i povratne pozive (callbacks). Nakon treniranja, sprema naučeni model i vraća povijest treninga.

- `predict(model, image, labels, top_k=3)`: Ova funkcija izvodi predikcije koristeći naučeni model. Prima model (prethodno naučeni model), `image` (sliku), listu `labels` (oznake) i opcionalni parametar `top_k` (broj najboljih predikcija koje treba vratiti, pretpostavljeno je 3). Predobradjuje sliku koristeći funkciju `preprocess`, proširuje dimenzije slike i prolazi kroz model koristeći metodu `predict`. Dohvaća najbolje predikcije i njihove odgovarajuće indekse, dohvaća oznake za te indekse i vraća odgovarajuće oznake i predikcije.

- `load_image(image_url, target_size)`: Ova funkcija učitava sliku sa zadane adrese i preoblikuje je na odgovarajuću veličinu.

- `make_predictions(image_url=None)`: Ova funkcija vrši predikcije koristeći naučeni model i prikazuje rezultate. Prihvaća opcionalni parametar `image_url`, koji pretpostavljeno ima vrijednost `https://httpstatusdogs.com/404-not-found` ako nije naveden. Poziva funkciju `load_image` za učitavanje slike s određenog `image_url` i promjenu veličine. Zatim poziva funkciju `predict` za dobivanje najboljih oznaka i predikcija za sliku koristeći naučeni model i zadane `labels`. Prikazuje sliku koristeći `imshow` iz biblioteke Matplotlib, prikazuje najbolje oznake i predikcije te ih ispisuje.

Tada moramo definirati još nekoliko varijabli:

```python
timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = "/content/drive/MyDrive/dataset"
checkpoint_path1 = "/content/drive/MyDrive/dataset/checkpoints/mobilenet" + timestamp_str + "/model.ckpt"
checkpoint_dir1 = os.path.dirname(checkpoint_path1)
log_create_dir1 = os.path.dirname(save_path + "/logs/fit/mobilenet/")
log_dir1 = "/content/drive/MyDrive/dataset/logs/fit/mobilenet/" + timestamp_str
```

U ovom dijelu koda, varijabli `timestamp_str` dodjeljuje se format datuma koristeći metodu `strftime` iz modula `datetime`. Varijabla `save_path` postavljena je na `/content/drive/MyDrive/dataset`, što predstavlja direktorij u kojem će se spremiti skup podataka.

Varijabla `checkpoint_path1` konstruira se dodavanjem `timestamp_str` na putanju direktorija za kontrolne točke, `/content/drive/MyDrive/dataset/checkpoints/mobilenet`, nakon čega slijedi ime checkpointa `/model.ckpt`. Varijabli `checkpoint_dir1` zatim se dodjeljuje putanja direktorija izdvojena iz `checkpoint_path1` koristeći funkciju `os.path.dirname`.

Slično tome, varijabli `log_create_dir1` dodjeljuje se putanja direktorija izdvojena iz konkatenacije `save_path` i `/logs/fit/mobilenet/`. Konačno, varijabli `log_dir1` dodjeljuje se potpuna putanja za direktorij s log datotekama dodavanjem `timestamp_str` na "/content/drive/MyDrive/dataset/logs/fit/mobilenet/".

U ovom je trenutku sve spremno za kreiranje i treniranje modela:

```python
mobilenet = create_model()
training(mobilenet, 'mobileNet_dogs', log_dir1, checkpoint_path1)
evaluate(mobilenet)
```

U ovom dijelu koda, najprije se stvara model pozivom funkcije `create_model()` i dodjeljuje se varijabli `mobilenet`. Zatim se navedeni model trenira pozivom funkcije `training()`, koja prima model, naziv modela za identifikaciju, putanju do direktorija za spremanje logova i putanju do kontrolne točke modela. Nakon treniranja, model se evaluira pozivom funkcije `evaluate()` kako bi se dobila mjera performansi na testnim podacima. Ovaj dio koda obuhvaća stvaranje, treniranje i evaluaciju modela te pridružene operacije poput spremanja logova i kontrolnih točaka.

## Rezultati i primjeri predikcija

Pozivanjem funkcije `evaluate()` možemo dobiti evaluaciju istreniranog modela:

```python
evaluate(mobilenet)

269/269 [==============================] - 11s 41ms/step - loss: 0.8559 - accuracy: 0.7621
loss: 0.86
accuracy: 0.76
```

Ovi metrički rezultati pružaju procjenu performansi modela. Vrijednost gubitka (loss) od 0.86 označava prosječnu pogrešku modelovih predikcija, pri čemu su niže vrijednosti poželjnije. Što se tiče točnosti (accuracy), model postiže rezultat od 0.76, što znači da ispravno predviđa klasu slike otprilike 76% vremena.

Od generiranih logova tijekom treniranja, možemo izvući razne metrike i grafove korištenjem `Tensorboarda`:

```
notebook.start("--logdir=/content/drive/MyDrive/dataset/logs/fit/mobilenet")
```

![acc](images/epoch_acc.jpg)

![loss](images/epoch_loss.jpg)

Na sljedeći način možemo generirati i klasifikacijski izvještaj:

```
target_names = load_labels()
model = load_model('/content/drive/MyDrive/dataset/mobileNet_dogs.h5')
validation_set = prepare(ds_test, INPUT_SHAPE, NUM_BREEDS, batch_size=BATCH_SIZE)
true_labels = []
predicted_labels = []
for image, label in validation_set:
    true_labels.extend(np.argmax(label, axis=1))
    prediction = model.predict(image)
    predicted_labels.extend(np.argmax(prediction, axis=1))
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)
print(classification_report(true_labels, predicted_labels, target_names=target_names))
```

|             Class              | Precision | Recall | F1-Score | Support |
| :----------------------------: | :-------: | :----: | :------: | :-----: |
|           chihuahua            |   0.56    |  0.75  |   0.64   |   52    |
|        japanese_spaniel        |   0.93    |  0.87  |   0.90   |   85    |
|          maltese_dog           |   0.84    |  0.86  |   0.85   |   152   |
|            pekinese            |   0.71    |  0.71  |   0.71   |   49    |
|            shih-tzu            |   0.67    |  0.64  |   0.65   |   114   |
|        blenheim_spaniel        |   0.90    |  0.86  |   0.88   |   88    |
|            papillon            |   0.91    |  0.89  |   0.90   |   96    |
|          toy_terrier           |   0.67    |  0.64  |   0.65   |   72    |
|      rhodesian_ridgeback       |   0.66    |  0.53  |   0.58   |   72    |
|          afghan_hound          |   0.90    |  0.88  |   0.89   |   139   |
|             basset             |   0.84    |  0.79  |   0.81   |   75    |
|             beagle             |   0.83    |  0.72  |   0.77   |   95    |
|           bloodhound           |   0.93    |  0.76  |   0.84   |   87    |
|            bluetick            |   0.73    |  0.77  |   0.75   |   71    |
|    black-and-tan_coonhound     |   0.77    |  0.73  |   0.75   |   59    |
|          walker_hound          |   0.54    |  0.58  |   0.56   |   53    |
|        english_foxhound        |   0.70    |  0.74  |   0.72   |   57    |
|            redbone             |   0.58    |  0.73  |   0.65   |   48    |
|             borzoi             |   0.78    |  0.84  |   0.81   |   51    |
|        irish_wolfhound         |   0.74    |  0.62  |   0.68   |   118   |
|       italian_greyhound        |   0.65    |  0.67  |   0.66   |   82    |
|            whippet             |   0.68    |  0.62  |   0.65   |   87    |
|          ibizan_hound          |   0.79    |  0.82  |   0.80   |   88    |
|       norwegian_elkhound       |   0.93    |  0.89  |   0.91   |   96    |
|           otterhound           |   0.68    |  0.88  |   0.77   |   51    |
|             saluki             |   0.85    |  0.78  |   0.81   |   100   |
|       scottish_deerhound       |   0.79    |  0.76  |   0.78   |   132   |
|           weimaraner           |   0.84    |  0.88  |   0.86   |   60    |
|   staffordshire_bullterrier    |   0.59    |  0.47  |   0.53   |   55    |
| american_staffordshire_terrier |   0.42    |  0.56  |   0.48   |   64    |
|       bedlington_terrier       |   0.91    |  0.96  |   0.93   |   82    |
|         border_terrier         |   0.79    |  0.78  |   0.78   |   72    |
|       kerry_blue_terrier       |   0.82    |  0.84  |   0.83   |   79    |
|         irish_terrier          |   0.65    |  0.70  |   0.67   |   69    |
|        norfolk_terrier         |   0.62    |  0.68  |   0.65   |   72    |
|        norwich_terrier         |   0.64    |  0.68  |   0.66   |   85    |
|       yorkshire_terrier        |   0.51    |  0.53  |   0.52   |   64    |
|    wire-haired_fox_terrier     |   0.84    |  0.75  |   0.80   |   57    |
|        lakeland_terrier        |   0.58    |  0.65  |   0.61   |   97    |
|        sealyham_terrier        |   0.93    |  0.95  |   0.94   |   102   |
|            airedale            |   0.85    |  0.79  |   0.82   |   102   |
|             cairn              |   0.84    |  0.71  |   0.77   |   97    |
|       australian_terrier       |   0.71    |  0.67  |   0.69   |   96    |
|         dandie_dinmont         |   0.97    |  0.93  |   0.95   |   80    |
|          boston_bull           |   0.91    |  0.78  |   0.84   |   82    |
|      miniature_schnauzer       |   0.77    |  0.67  |   0.71   |   54    |
|        giant_schnauzer         |   0.71    |  0.72  |   0.71   |   57    |
|       standard_schnauzer       |   0.61    |  0.65  |   0.63   |   55    |
|         scotch_terrier         |   0.82    |  0.81  |   0.82   |   58    |
|        tibetan_terrier         |   0.75    |  0.61  |   0.67   |   106   |
|         silky_terrier          |   0.74    |  0.58  |   0.65   |   83    |
|  soft-coated_wheaten_terrier   |   0.70    |  0.66  |   0.68   |   56    |
|  west_highland_white_terrier   |   0.79    |  0.91  |   0.85   |   69    |
|             lhasa              |   0.50    |  0.62  |   0.55   |   86    |
|     flat-coated_retriever      |   0.62    |  0.88  |   0.73   |   52    |
|     curly-coated_retriever     |   0.82    |  0.82  |   0.82   |   51    |
|        golden_retriever        |   0.80    |  0.78  |   0.79   |   50    |
|       labrador_retriever       |   0.82    |  0.70  |   0.76   |   71    |
|    chesapeake_bay_retriever    |   0.71    |  0.81  |   0.76   |   67    |
|  german_short-haired_pointer   |   0.73    |  0.69  |   0.71   |   52    |
|             vizsla             |   0.69    |  0.78  |   0.73   |   54    |
|         english_setter         |   0.77    |  0.67  |   0.72   |   61    |
|          irish_setter          |   0.81    |  0.87  |   0.84   |   55    |
|         gordon_setter          |   0.77    |  0.77  |   0.77   |   53    |
|        brittany_spaniel        |   0.82    |  0.69  |   0.75   |   52    |
|            clumber             |   0.79    |  0.84  |   0.82   |   50    |
|        english_springer        |   0.83    |  0.85  |   0.84   |   59    |
|     welsh_springer_spaniel     |   0.78    |  0.80  |   0.79   |   50    |
|         cocker_spaniel         |   0.73    |  0.73  |   0.73   |   59    |
|         sussex_spaniel         |   0.90    |  0.90  |   0.90   |   51    |
|      irish_water_spaniel       |   0.80    |  0.80  |   0.80   |   50    |
|             kuvasz             |   0.79    |  0.74  |   0.76   |   50    |
|           schipperke           |   0.72    |  0.85  |   0.78   |   54    |
|          groenendael           |   0.85    |  0.78  |   0.81   |   50    |
|            malinois            |   0.81    |  0.78  |   0.80   |   50    |
|             briard             |   0.70    |  0.77  |   0.73   |   52    |
|             kelpie             |   0.61    |  0.58  |   0.60   |   53    |
|            komondor            |   0.86    |  0.94  |   0.90   |   54    |
|      old_english_sheepdog      |   0.76    |  0.86  |   0.80   |   69    |
|       shetland_sheepdog        |   0.73    |  0.72  |   0.73   |   57    |
|             collie             |   0.63    |  0.68  |   0.65   |   53    |
|         border_collie          |   0.75    |  0.72  |   0.73   |   50    |
|      bouvier_des_flandres      |   0.72    |  0.82  |   0.77   |   50    |
|           rottweiler           |   0.79    |  0.88  |   0.84   |   52    |
|        german_shepherd         |   0.79    |  0.79  |   0.79   |   52    |
|            doberman            |   0.64    |  0.78  |   0.70   |   50    |
|       miniature_pinscher       |   0.79    |  0.76  |   0.78   |   84    |
|   greater_swiss_mountain_dog   |   0.72    |  0.72  |   0.72   |   68    |
|      bernese_mountain_dog      |   0.85    |  0.92  |   0.89   |   118   |
|          appenzeller           |   0.52    |  0.65  |   0.57   |   51    |
|          entlebucher           |   0.88    |  0.69  |   0.77   |   102   |
|             boxer              |   0.65    |  0.59  |   0.62   |   51    |
|          bull_mastiff          |   0.79    |  0.75  |   0.77   |   56    |
|        tibetan_mastiff         |   0.77    |  0.77  |   0.77   |   52    |
|         french_bulldog         |   0.78    |  0.78  |   0.78   |   59    |
|           great_dane           |   0.70    |  0.66  |   0.68   |   56    |
|         saint_bernard          |   0.89    |  0.96  |   0.92   |   70    |
|           eskimo_dog           |   0.35    |  0.44  |   0.39   |   50    |
|            malamute            |   0.61    |  0.72  |   0.66   |   78    |
|         siberian_husky         |   0.69    |  0.46  |   0.55   |   92    |
|         affenpinscher          |   0.73    |  0.80  |   0.76   |   50    |
|            basenji             |   0.92    |  0.79  |   0.85   |   109   |
|              pug               |   0.79    |  0.85  |   0.82   |   100   |
|            leonberg            |   0.91    |  0.88  |   0.89   |   110   |
|          newfoundland          |   0.73    |  0.75  |   0.74   |   95    |
|         great_pyrenees         |   0.78    |  0.77  |   0.77   |   113   |
|            samoyed             |   0.86    |  0.96  |   0.90   |   118   |
|           pomeranian           |   0.96    |  0.87  |   0.92   |   119   |
|              chow              |   0.98    |  0.94  |   0.96   |   96    |
|            keeshond            |   0.92    |  0.95  |   0.93   |   58    |
|       brabancon_griffon        |   0.76    |  0.85  |   0.80   |   53    |
|            pembroke            |   0.79    |  0.78  |   0.78   |   81    |
|            cardigan            |   0.58    |  0.65  |   0.62   |   55    |
|           toy_poodle           |   0.61    |  0.49  |   0.54   |   51    |
|        miniature_poodle        |   0.44    |  0.44  |   0.44   |   55    |
|        standard_poodle         |   0.73    |  0.64  |   0.68   |   59    |
|        mexican_hairless        |   0.81    |  0.95  |   0.87   |   55    |
|             dingo              |   0.70    |  0.68  |   0.69   |   56    |
|             dhole              |   0.80    |  0.94  |   0.86   |   50    |
|      african_hunting_dog       |   0.91    |  0.99  |   0.94   |   69    |
|            Accuracy            |     -     |   -    |   0.76   |  8580   |
|           Macro avg            |   0.75    |  0.76  |   0.75   |  8580   |
|          Weighted avg          |   0.77    |  0.76  |   0.76   |  8580   |

Na temelju izvještaja o klasifikaciji i matrice zabune, možemo izvući nekoliko ključnih zaključaka. Pasmina s najvećom preciznošću, odnosno najnižom stopom lažnih pozitiva, je Dandie Dinmont. Dakle, kada model predvidi da je pas Dandie Dinmont, obično je u pravu. S druge strane, pasmina s najnižom preciznošću, odnosno većom stopom lažnih pozitiva, je Chihuahua. Stoga, model često pogrešno klasificira neke druge pasmine kao Chihuahue.

Kada je riječ o odzivu, odnosno sposobnosti ispravnog prepoznavanja pozitivnih instanci, izdvaja se pasmina Entlebucher s najvišim rezultatom. To ukazuje na to da model rijetko propušta prepoznati Entlebuchera kada se pojavi na slikama. S druge strane, japanski spanijel ima najniži rezultat odziva, što sugerira da model često ne uspijeva ispravno identificirati tu pasminu.

Za procjenu ukupne uspješnosti modela možemo pogledati metriku točnosti, koja mjeri postotak točno predviđenih instanci za sve pasmine. U ovom slučaju, model postiže točnost od 82%, što ukazuje da pravilno klasificira pse u većini slučajeva.

![prva](images/labrador.jpg)

![druga](images/vizla.jpg)

![treca](images/amor.jpg)

![cetvrta](images/miks_dog.jpg)

![peta](images/baski_dog.jpg)

![sesta](images/maltezer.jpg)

![sedma](images/weimar.jpg)

## Literatura

- [1] <https://www.tensorflow.org/datasets/catalog/stanford_dogs>
