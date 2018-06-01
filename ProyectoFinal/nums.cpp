#include<bits/stdc++.h>
using namespace std;
#define LLI long long int

int main(){
    LLI n;
    cin >> n;
    cout << n << endl;
    for(LLI i=0; i<n; i++){
        cout << rand()%1000 << endl;
    }
    return 0;
}
