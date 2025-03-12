## 1 Wymagania projektu

W ramach projektu należy stworzyć program, który będzie realizował opisane w
temacie funkcje. Projekt jest zadaniem zespołowym, gdzie każdy zespół składa
się z 2 osób.
Głównym językiem programowania powinien być język Python wraz z fra-
meworkiem przeznaczonym do sieci neuronowych: Pytorch/Tensorflow.
Za projekt można uzyskać maksymalnie x × 10p., gdzie x to liczba osób w
zespole. Każdy z członków zespołu może dostać maksymalnie 10 punktów.
Ocenie w ramach projektu podlegają:
1. Działanie programu - realizacja funkcji (7 p.)
2. Dokumentacja dokonanych eksperymentów oraz wizualizacja wyników (3
p.)

Projekt uznaje się za oddany w momencie prezentacji go prowadzącemu

## 2 Modyfikacja obrazów

W ramach projektu należy zaproponować architekturę oraz wykorzystać ją do
realizacji 2 z 4 zadań opisanych poniżej. W celu badania jakości stworzonego roz-
wiązania należy zastosować wszystkie metryki opisane w ramach wykładów tzn.
SNE, PSNR, SSIM, LPIPS. Należy dążyć do rozwiązania lepszego niż bazowe
zaproponowane w zadaniach.
Do każdego z zadań korzystamy ze zbioru danych DIV2K: https://data.
vision.ee.ethz.ch/cvl/DIV2K/.
Zadanie oznaczone * jest uznawane za trudniejsze od pozostałych. Wykona-
nie go umożliwia późniejszą zmianę ścieżki projektowej na ścieżkę badawczą.
W ramach trenowania i walidacji wyników eksperymentów, korzystamy z
obrazów w rozdzielczości 256 × 256.

### 2.3 Deblurowanie

Parametry danych wejściowych: wielkość kernela dla rozmycia gaussow-
skiego: 3 × 3, 5 × 5.
Przygotowanie zbioru danych: należy napisać metodę do rozmywania gaus-
sowskiego zdjęć np. korzystając z metody GaussianBlur z biblioteki OpenCV

### 2.4 Inpainting*

Parametry danych wejściowych: losowe wycinanie obszarów o wielkościach:
3 × 3, 32 × 32.
Przygotowanie zbioru danych: należy napisać metodę do losowego wyci-
nania fragmentów obrazów.
Porównanie wyników: metodą bazową z którą należy porównać stworzone
rozwiązanie jest inpaint z OpenCV z parametrem INPAINT TELEA.
