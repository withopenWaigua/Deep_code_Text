#include<iostream>
using namespace std;
#include<vector>
int minCost(vector<int> Cost) 
{
 int len = Cost.size();
    if (len == 2) {
        return Cost[0] < Cost[1] ? Cost[0] : Cost[1];
    }
    else
    {
        for (int i = len - 3; i >= 0; i--) 
        {
            Cost[i] = Cost[i] + min(Cost[i + 1], Cost[i + 2]);
        }
            return min(Cost[0], Cost[1]);
    }
}
int main() {
    vector<int> cost = {5,3,30,80,60};       //在此输入一个数组
    int num =minCost(cost);
    cout << num << endl;
    return 0; 
}
