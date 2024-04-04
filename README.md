**Project: Rainfall Prediction in Bangkok**

**Description**

This project aims to predict rainfall in Bangkok using a linear regression model. The model is trained on historical rainfall data collected from the Bangkok website.

**Data**

The data used in this project was scraped from the Bangkok website. The data contains rainfall measurements for 311,211 rows, from 02/01/2564 08:20 to 18/02/2566 16:50. The data is divided by district, with the following number of rows for each district:

| District    | Number of Rows |
| ----------- | -------------- |
| หนองจอก     | 16679          |
| มีนบุรี     | 12076          |
| คลองเตย     | 11711          |
| บางซื่อ     | 11186          |
| บางแค       | 9942           |
| ทวีวัฒนา    | 9520           |
| ราชเทวี     | 9177           |
| พระโขนง     | 9156           |
| บางขุนเทียน | 8921           |
| จตุจักร     | 8773           |
| ภาษีเจริญ   | 8770           |
| สายไหม      | 8748           |
| บางนา       | 8330           |
| วังทองหลาง  | 8254           |
| พระนคร      | 8073           |
| ตลิ่งชัน    | 7739           |
| ประเวศ      | 7678           |
| ยานนาวา     | 7603           |
| จอมทอง      | 7227           |
| ธนบุรี      | 7125           |
| หลักสี่     | 6397           |
| สะพานสูง    | 5959           |
| วัฒนา       | 5863           |
| ดุสิต       | 5707           |
| ห้วยขวาง    | 5594           |
| ปทุมวัน     | 5525           |
| บางกอกน้อย  | 5453           |
| คลองสาน     | 5390           |
| สาทร        | 5324           |
| บางรัก      | 5249           |
| บางคอแหลม   | 5196           |
| ราษฏร์บูรณะ | 5076           |
| คลองสามวา   | 4990           |
| ทุ่งครุ     | 4766           |
| ลาดกระบัง   | 4502           |
| สวนหลวง     | 3647           |
| บางเขน      | 3475           |
| บางกอกใหญ่  | 3449           |
| หนองแขม     | 3276           |
| พญาไท       | 3245           |
| ดินแดง      | 3048           |
| คันนายาว    | 2982           |
| ป้อมปราบฯ   | 2922           |
| บางบอน      | 2860           |
| บึงกุ่ม     | 2853           |
| บางกะปิ     | 2790           |
| สัมพันธวงศ์ | 2775           |
| บางพลัด     | 2731           |
| ดอนเมือง    | 2498           |
| ลาดพร้าว    | 981            |

**Installation**

1. Create a virtual environment:

```
python -m venv env
```

2. Activate the environment:

- **Windows:**

```
env\Scripts\activate
```

- **Mac:**

```
source env/bin/activate
```

3. Install the requirements using pip:

```
pip install -r requirements.txt
```

4. Run the `train.py` file:

```
python train.py
```

**Experiment**

The project uses a linear regression model to predict rainfall. The model is trained using the historical rainfall data and evaluated using the mean squared error (MSE). The MSE is calculated as follows:

```
MSE = np.mean((self.y - self.x @ w)**2)
```

The weights of the model are updated using the following formula:

```
w = w + (-1 * lr * gradient(w))
```

The gradient of the MSE with respect to the weights is calculated as follows:

```
grad_v[:-1] = -2 / self.x.shape[0] * np.sum(self.x[:, :-1] * errors[:, None], axis=0)
grad_v[-1]  = -2 / self.x.shape[0] * np.sum(predictions)
```

The weights of the model are updated using the following equation:

```python
w = w + (-1 * lr * gradient(w))
```
