struct VSOutput
{
    float4 position : SV_Position; // window space position for rasterization
    float4 color : COLOR; // color passed to pixel shader
};

float4 main(VSOutput input) : SV_Target 
{
    // output the color passed from the vertex shader
    return input.color;
}