#include <bits/stdc++.h>

using namespace std;

int main(){
    int f1, f2, c1, c2;
    float num;
    cin >> f1 >> c1;
    cin >> f2 >> c2;
    cout << f1 << " " << c1 << endl;
    cout << f1 << " " << c2 << endl;
    for(int i=0; i<f1; i++){
        for(int j=0; j<c1; j++){
            num = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            cout << num << " ";
        }
        cout << endl;
    }
    cout << endl << endl;
    for(int i=0; i<f2; i++){
        for(int j=0; j<c2; j++){
            num = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            cout << num << " ";
        }
        cout << endl;
    }
    return 0;
}