#version 310 es
#undef highp

in highp vec4 vColor;
out highp vec4 fColor;

void main()
{
   fColor = vColor;
}
