# Chapter 1: Heat Transfer

This chapter introduces the three ways heat moves from one place to another: conduction, convection and radiation. It sets out the basic laws for each, works through simple examples with numbers, and shows how the three combine in everyday objects such as a wall, a cup of tea and a pump motor.

By the end of the chapter you should be able to tell which mode of heat transfer matters most in a given situation, write down the rate equation for it, and estimate a heat flow to within a factor of two.

## 1.1 Heat and temperature

**Heat** is energy in transit because of a temperature difference. It is not something a body contains; a body contains internal energy, and heat is the name for the part of that energy that crosses its boundary because the body is hotter or colder than its surroundings.

**Temperature** is a measure of how hot a body is. Two bodies at the same temperature exchange no net heat when they touch. Temperature differences drive heat flow in the same way that pressure differences drive the flow of water in a pipe.

In this chapter heat is measured in joules (J) and the rate of heat flow in watts (W), where one watt is one joule per second. Temperature is measured in kelvin (K) or degrees Celsius; a temperature difference of one kelvin is the same as a difference of one degree Celsius, so either unit can be used wherever only a difference appears.

**Heat flux** is the rate of heat flow per unit area, measured in watts per square metre. It is written q, while the total rate of heat flow through an area A is written Q, so that Q = q A.

Heat always flows from the hotter body to the colder one unless work is done to move it the other way. This is one statement of the second law of thermodynamics, and it is why a refrigerator needs a power supply.

## 1.2 Conduction

**Conduction** is the transfer of heat through a material without any bulk movement of the material. In a solid, it happens because faster vibrating atoms pass energy to their slower neighbours, and in metals the free electrons carry most of it.

The rate of conduction through a flat slab is given by Fourier's law. For a slab of thickness L and area A, with faces held at temperatures T1 and T2, the rate of heat flow is Q = k A (T1 - T2) / L, where k is the thermal conductivity of the material.

**Thermal conductivity** is a property of a material that says how easily heat passes through it, measured in watts per metre per kelvin (W/m K). Metals have high values, gases have low values, and most building materials lie in between.

| Material | Thermal conductivity (W/m K) |
| --- | --- |
| Copper | 390 |
| Aluminium | 210 |
| Steel | 45 |
| Glass | 1.0 |
| Brick | 0.7 |
| Water | 0.6 |
| Wood (pine) | 0.12 |
| Mineral wool | 0.04 |
| Still air | 0.025 |

Figure 1.1: Heat flow through a flat slab. The temperature falls in a straight line from the hot face to the cold face when k does not change with temperature.

The straight line in Figure 1.1 is worth remembering. When the conductivity is constant and no heat is produced inside the slab, the temperature at any point inside is found by simple proportion between the two face temperatures.

### Worked example 1.1: Heat loss through a window

A single glazed window is 1.2 m wide, 1.0 m high and 4 mm thick. The inner surface of the glass is at 12 degrees Celsius and the outer surface at 10 degrees Celsius. Find the rate of heat loss by conduction through the glass.

The area is A = 1.2 x 1.0 = 1.2 square metres, the thickness is L = 0.004 m and the conductivity of glass is k = 1.0 W/m K from the table. Then Q = 1.0 x 1.2 x (12 - 10) / 0.004 = 600 W.

A temperature drop of only two degrees across the glass already lets 600 W through. In practice most of the resistance to heat loss through a window lies in the thin layers of air on each side of the glass, not in the glass itself; section 1.5 shows how to account for them.

## 1.3 Convection

**Convection** is the transfer of heat between a surface and a moving fluid, such as air or water. The fluid next to the surface is heated or cooled by conduction, and the movement of the fluid carries that energy away.

Convection is called natural or free when the fluid moves only because warmer fluid is lighter and rises, as in the air above a radiator. It is called forced when a fan, a pump or the wind drives the fluid, as in the air blown over a motor by its cooling fan.

The rate of convection from a surface is given by Newton's law of cooling: Q = h A (Ts - Tf), where Ts is the surface temperature, Tf is the temperature of the fluid far from the surface, A is the area and h is the heat transfer coefficient.

The **heat transfer coefficient** h is not a property of a material. It depends on the fluid, its speed, the shape of the surface and whether the flow is smooth or turbulent. Typical values are 2 to 25 W/m2 K for natural convection in air, 25 to 250 W/m2 K for forced convection in air and 100 to 20000 W/m2 K for water.

Figure 1.2: Velocity and temperature near a heated plate in a moving fluid. Both change sharply within a thin layer next to the surface, called the boundary layer.

The thin layer shown in Figure 1.2 explains why h rises with fluid speed. A faster flow makes the boundary layer thinner, and a thinner layer offers less resistance to the heat crossing it.

### Worked example 1.2: Cooling a motor housing

A small pump motor has a housing with an outer area of 0.15 square metres. Its surface is at 65 degrees Celsius and the room air is at 25 degrees Celsius. The fan blows air over the housing, giving h = 40 W/m2 K. How much heat does the housing lose by convection?

Q = h A (Ts - Tf) = 40 x 0.15 x (65 - 25) = 240 W.

If the fan cover is blocked by dust, the flow falls and the coefficient might drop to about 10 W/m2 K, typical of natural convection. To lose the same 240 W, the temperature difference would then have to rise to 240 / (10 x 0.15) = 160 K, which would put the housing at 185 degrees Celsius. Long before that, the motor's thermal protection would switch it off. Keeping cooling air paths clear matters.

## 1.4 Radiation

**Thermal radiation** is energy carried by electromagnetic waves emitted by every body because of its temperature. Unlike conduction and convection, it needs no material to travel through, which is how the sun heats the earth across empty space.

The most a surface can radiate at a given temperature is given by the Stefan-Boltzmann law. A perfect emitter, called a black body, radiates q = sigma T^4 per unit area, where T is the absolute temperature in kelvin and sigma = 5.67 x 10^-8 W/m2 K4 is the Stefan-Boltzmann constant.

**Emissivity** is the ratio of the radiation emitted by a real surface to that from a black body at the same temperature. It lies between 0 and 1. Polished metals have low emissivity, around 0.05, while paint, brick, skin and most non-metals have high emissivity, around 0.9.

For a small body of area A, emissivity e and temperature Ts inside a large enclosure at temperature Tsur, the net rate of radiation is Q = e sigma A (Ts^4 - Tsur^4). Both temperatures must be in kelvin.

Figure 1.3: Net radiation between a small body and large surroundings. The body emits according to its own temperature and absorbs according to the temperature of the surroundings.

### Worked example 1.3: Radiation from a person

A person with a skin and clothing surface area of 1.8 square metres and an emissivity of 0.95 has a surface temperature of 30 degrees Celsius (303 K). The walls of the room are at 18 degrees Celsius (291 K). Estimate the net heat loss by radiation.

Q = 0.95 x 5.67 x 10^-8 x 1.8 x (303^4 - 291^4). The bracket is 8.43 x 10^9 - 7.17 x 10^9 = 1.26 x 10^9 K4, so Q is about 122 W.

That is of the same size as the heat a resting person produces, around 100 W. This is why a room with cold walls feels chilly even when the air in it is warm.

When the temperature difference is small compared with the absolute temperature, radiation can be written in the same form as convection, Q = hr A (Ts - Tsur), with a radiation coefficient hr of about 4 e sigma T^3. Near room temperature hr is about 5 to 6 W/m2 K for a high emissivity surface.

## 1.5 Thermal resistance

**Thermal resistance** is the temperature difference needed to drive one watt of heat through a path, measured in kelvin per watt. It allows heat flow problems to be solved in the same way as electrical circuits, with temperature difference in place of voltage and heat flow in place of current.

For a slab, the conduction resistance is R = L / (k A). For a surface, the convection resistance is R = 1 / (h A). The heat flow through any path is then Q = (T1 - T2) / R.

Resistances in series add. When heat passes in turn through an inside air layer, a wall and an outside air layer, the total resistance is the sum of the three, and the heat flow is the overall temperature difference divided by that sum.

Resistances in parallel combine like electrical resistors in parallel: the reciprocal of the total is the sum of the reciprocals. A wall with a window in it is a parallel path: some heat goes through the wall and some through the glass.

Figure 1.4: A wall drawn as three thermal resistances in series, from the inside air to the outside air.

### Worked example 1.4: Window with air films

Return to the window of worked example 1.1. Now the room air is at 20 degrees Celsius and the outside air at 0 degrees Celsius. The inside film coefficient is 8 W/m2 K and the outside one is 25 W/m2 K. Find the heat loss.

The inside film resistance is 1 / (8 x 1.2) = 0.104 K/W. The glass resistance is 0.004 / (1.0 x 1.2) = 0.0033 K/W. The outside film resistance is 1 / (25 x 1.2) = 0.033 K/W. The total is 0.141 K/W.

Q = (20 - 0) / 0.141 = 142 W.

The glass contributes only about 2 percent of the total resistance. Double glazing works mainly because it adds a layer of still air between the panes, not because it doubles the glass.

The **U-value** of a building element is the reciprocal of its total resistance per unit area, measured in W/m2 K. For the window above it is 1 / (0.141 x 1.2), about 5.9 W/m2 K. Building rules usually set a largest allowed U-value for walls, roofs and windows.

## 1.6 Combined modes

Real objects lose heat by more than one mode at once. A hot pipe in a room loses heat by natural convection to the air and by radiation to the walls, and the two act in parallel from the same surface. The total heat loss is the sum of the two, or in resistance terms the two resistances combine in parallel.

When the surface temperature is close to room temperature, it is often simplest to add the convection and radiation coefficients into one combined surface coefficient. For a painted surface in still room air, a combined coefficient of about 10 W/m2 K is a useful first estimate.

Figure 1.5: Heat leaving a hot pipe by convection to the air and by radiation to the walls. The two paths start at the same surface temperature.

### Worked example 1.5: Insulating a hot water pipe

A bare steel pipe carries water at 60 degrees Celsius through a room at 20 degrees Celsius. Its outer diameter is 28 mm, so each metre has an outer area of about 0.088 square metres. Using a combined coefficient of 10 W/m2 K, the loss per metre is 10 x 0.088 x 40, about 35 W.

Wrapping the pipe with 20 mm of mineral wool adds a conduction resistance far larger than the surface resistance. A simple estimate treats the wrap as a flat layer of mean area 0.15 square metres per metre: R = 0.02 / (0.04 x 0.15) = 3.3 K/W. The outer surface of the wrap has an area of about 0.21 square metres and a resistance of 1 / (10 x 0.21) = 0.48 K/W.

The total is about 3.8 K/W, so the loss falls to 40 / 3.8, about 10.5 W per metre, less than a third of the bare pipe loss.

## 1.7 Choosing the dominant mode

It is rarely necessary to compute every mode exactly. A rough check of which mode dominates saves effort and points to the right fix when something runs too hot or too cold.

Conduction dominates inside solids and through thin layers of good conductors. Forced convection dominates wherever a fan or pump moves a fluid quickly over a surface. Radiation becomes important when surfaces are very hot, when there is a vacuum, or when the convection is weak, as with still air at room temperature, where radiation and natural convection are roughly equal.

> As a rule of thumb, if doubling the fluid speed would change the answer a lot, convection is the mode to study first.

## 1.8 Heating and cooling over time

So far every example has been steady: temperatures did not change with time. Many practical questions are about how long something takes to warm up or cool down, such as a cup of tea left on a desk or a motor after it is switched off.

**Heat capacity** is the energy needed to raise the temperature of a body by one kelvin, measured in joules per kelvin. It equals the mass of the body times the specific heat capacity of its material. Water has a specific heat capacity of about 4180 J/kg K, steel about 490 J/kg K and aluminium about 900 J/kg K.

When a body is small or a good conductor, its inside stays at nearly one temperature while it cools. This is called the **lumped capacity** model. The body then cools so that the difference between its temperature and the surroundings falls by the same fraction in each equal interval of time.

The **time constant** of a cooling body is its heat capacity divided by h A, the product of the surface coefficient and the area. After one time constant the temperature difference has fallen to about 37 percent of its starting value; after three time constants it has fallen to about 5 percent.

Figure 1.6: Temperature of a cooling body against time. The curve falls steeply at first and flattens as the body approaches room temperature.

The lumped model is good when the Biot number, h L / k with L the volume divided by the surface area, is below about 0.1. For a metal part in air this is almost always true. For a thick block of wood or a large volume of still water it is not, and the inside lags behind the surface.

### Worked example 1.6: Cooling a cup of tea

A cup holds 0.25 kg of tea at 80 degrees Celsius in a room at 20 degrees Celsius. The cup and tea lose heat from an area of 0.03 square metres with a combined coefficient of 12 W/m2 K. Estimate how long the tea takes to reach 50 degrees Celsius, ignoring the cup itself.

The heat capacity is 0.25 x 4180 = 1045 J/K. The time constant is 1045 / (12 x 0.03) = 2900 s, about 48 minutes.

The temperature difference must fall from 60 K to 30 K, which is half. A difference halves after about 0.69 time constants, so the time is about 0.69 x 48, roughly 33 minutes.

In practice the tea cools faster at first because of evaporation from its open surface, which this simple model leaves out. A lid makes the model much closer to the truth.

The same steps can be written as a short program. The lines that start with a hash sign are comments, not headings:

```
# lumped cooling of the tea in worked example 1.6
C = 0.25 * 4180          # heat capacity, J/K
tau = C / (12 * 0.03)    # time constant, s
# time to halve the difference
t_half = 0.693 * tau
```

## 1.9 Heat exchangers in brief

A **heat exchanger** is a device that passes heat from one fluid to another without letting them mix, such as a car radiator or the coil inside a hot water cylinder. The two fluids flow on either side of a wall, and the heat crosses the wall by convection, conduction and convection again.

The resistance of the path is found as in section 1.5, as three resistances in series. The heat flow is then the temperature difference between the fluids divided by that resistance, but because both fluids change temperature as they pass through, a mean temperature difference must be used. That calculation belongs to a later chapter.

Fluids can flow in the same direction, called parallel flow, or in opposite directions, called counterflow. Counterflow lets the cold fluid leave hotter than the hot fluid leaves, so for the same area it moves more heat.

Figure 1.7: Temperatures along a parallel flow and a counterflow exchanger. In counterflow the two curves stay roughly the same distance apart.

Key terms
---------

Heat, temperature, heat flux, conduction, thermal conductivity, convection, heat transfer coefficient, thermal radiation, emissivity, thermal resistance, U-value, heat capacity, lumped capacity, time constant, heat exchanger.

## Summary

- Heat is energy crossing a boundary because of a temperature difference; heat flux is the rate per unit area.
- Conduction follows Fourier's law, Q = k A (T1 - T2) / L, and depends on the thermal conductivity of the material.
- Convection follows Newton's law of cooling, Q = h A (Ts - Tf); the coefficient h depends on the flow, not just the material.
- Radiation follows the Stefan-Boltzmann law with absolute temperatures to the fourth power, reduced by the emissivity of the surface.
- Thermal resistances in series add, and in parallel combine by their reciprocals, just like electrical resistors.
- Thin air films often control the heat flow through windows and walls more than the solid layers do.
- A combined surface coefficient of about 10 W/m2 K is a useful first estimate for still room air.

## Exercises

1. A brick wall 0.2 m thick and 10 square metres in area has faces at 18 and 4 degrees Celsius. Find the conduction heat flow.
2. Explain why a metal spoon left in hot tea becomes hot at the handle while a wooden spoon does not.
3. A hot plate at 120 degrees Celsius loses 300 W by natural convection to air at 20 degrees Celsius. Estimate its area if h = 10 W/m2 K.
4. Repeat worked example 1.3 for a person wearing a coat whose outer surface is at 22 degrees Celsius.
5. A copper and a steel bar of equal size are held between the same two temperatures. Which carries more heat, and by what factor?
