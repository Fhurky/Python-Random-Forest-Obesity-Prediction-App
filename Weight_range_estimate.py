from time import sleep
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# CSV dosyasını oku
with open("Data_set.csv", "r") as file:
    lines = file.readlines()

# Her satırı virgülle ayırarak bir listeye dönüştür
data = [line.strip().split(",") for line in lines]

# Veri çerçevesini oluştur
df = pd.DataFrame(data, columns=["Age", "Gender", "Height", "Weight", "CALC", "FAVC", "FCVC", "NCP", "SCC", "SMOKE", "CH2O", "family_history_with_overweight", "FAF", "TUE", "CAEC", "MTRANS", "NObeyesdad"])

# Veri setini keşfet ve ön işlemleri yap
# Örneğin, eksik verileri doldurabilir, kategorik değişkenleri kodlayabilir, vb.

# Hedef değişkeni (NObeyesdad) ve özellikleri ayır
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

# Kategorik değişkenleri kodlayın
label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)

# Modeli seçin ve eğitin
model = RandomForestClassifier(n_estimators=4, random_state=42)
model.fit(X_encoded, y)

# Modelin performansını değerlendirin
y_pred = model.predict(X_encoded)
accuracy = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)

print("Modelin Performansı:")
print("Accuracy:", accuracy)
print("Classification Report:")
print(report)

# Veri setini eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Modelin performansını değerlendir
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Modelin Performansı:")
print("Accuracy:", accuracy)
print("Classification Report:")
print(report)


# İlk 10 kişinin tahminlerini al
first_10_predictions = model.predict(X_encoded.head(10))

print("\nEğitim Setindeki İlk 10 Kişinin Tahminleri:")
for i, prediction in enumerate(first_10_predictions):
    print(f"Kişi {i + 1} Tahmini: {prediction}")

# Rastgele kişi oluştur
new_person = pd.DataFrame({
    "Age": [23],  # Örnek bir yaş
    "Gender": ["Male"],  # Örnek bir cinsiyet
    "Height": [1.80],  # Örnek bir boy
    "Weight": [75],  # Örnek bir kilo (kilolu olduğunu belirten bir değer)
    "CALC": ["no"],  # Örnek bir değer
    "FAVC": ["no"],  # Örnek bir değer
    "FCVC": [2],  # Örnek bir değer
    "NCP": [3],  # Örnek bir değer
    "SCC": ["no"],  # Örnek bir değer
    "SMOKE": ["no"],  # Örnek bir değer
    "CH2O": [2],  # Örnek bir değer
    "family_history_with_overweight": ["yes"],  # Örnek bir değer
    "FAF": [0],  # Örnek bir değer
    "TUE": [1],  # Örnek bir değer
    "CAEC": ["Sometimes"],  # Örnek bir değer
    "MTRANS": ["Public_Transportation"]  # Örnek bir değer
})

# Eğitim verisindeki kategorik değişkenlerin sınıflarını al
label_encoder_classes = {}
for col in X.columns:
    if X[col].dtype == 'object':
        label_encoder = LabelEncoder()
        label_encoder.fit(X[col])
        label_encoder_classes[col] = label_encoder.classes_

# Yeni kişi verilerini kodlarken kullanılacak label_encoder_classes nesnesini kullanarak sınıfları belirleyin
new_person_encoded = new_person.copy()
for col, classes in label_encoder_classes.items():
    new_person_encoded[col] = new_person[col].apply(lambda x: np.where(classes == x)[0][0] if x in classes else -1)



# Tahmini sonucu ekrana yazdır
print("\nTahmini Sonuç:", prediction)